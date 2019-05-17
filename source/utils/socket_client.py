import queue
import multiprocessing
import os
import base64
import io
from imageio import imread
import imageio
from threading import Thread
from core.cv_utils import base64str_to_frame
from socketIO_client_nexus import SocketIO, BaseNamespace
# import socketIO_client_nexus.exceptions.ConnectionError
from config import Config
import time


class SocketError:

    Disconnect = 'SocketDisconnect'

def catch_disconnecting_with_init_func(func, init_func):

    def func_wrapper(*args, **kwargs):
        result = SocketError.Disconnect
        while result == SocketError.Disconnect:
            try:
                result = func(*args, **kwargs)
            except:
                try:
                    init_func()
                except:
                    pass
                print('Caught disconnecting')
                time.sleep(5)
        return result

    return func_wrapper


def catch_disconnecting(func, *args, **kwargs):

    def func_wrapper(*args, **kwargs):
        result = SocketError.Disconnect
        while result == SocketError.Disconnect:
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print('Caught something')
                print(e)
                time.sleep(5)

        return result

    return func_wrapper


def catch_disconnect_all_class_methods(Cls):

    class ClassWrapper(object):

        @catch_disconnecting
        def __init__(self, *args, **kwargs):
            self.oInstance = Cls(*args, **kwargs)

        def __getattribute__(self, s):
            """
            this is called whenever any attribute of a NewCls object is accessed. This function first tries to
            get the attribute off NewCls. If it fails then it tries to fetch the attribute from self.oInstance (an
            instance of the decorated class). If it manages to fetch the attribute from self.oInstance, and
            the attribute is an instance method then `time_this` is applied.
            """
            try:
                x = super(ClassWrapper, self).__getattribute__(s)
            except AttributeError:
                pass
            else:
                return x
            x = self.oInstance.__getattribute__(s)
            init_func = self.oInstance.__init__

            if type(x) == type(self.__init__): # it is an instance method
                return catch_disconnecting_with_init_func(x, init_func)                 # this is equivalent of just decorating the method with time_this
            else:
                return x

    return ClassWrapper


@catch_disconnect_all_class_methods
class ImageSocket():
    RESULT_SUCESSFUL = 'successful'
    RESULT_ALERT = 'alert'
    RESULT_ERROR = 'error'

    def __init__(self, hostname, port):
        imageio.plugins.freeimage.download()
        self.hostname = hostname
        self.port = port
        self.queue = queue.Queue()
        print('Connecting on %s:%s' %(self.hostname, self.port))
        self.socket = SocketIO(self.hostname, self.port)
        self.image_namespace = self.socket.define(ImageNamespace,
                                                  Config.Socket.IMAGE_NAMESPACE)
        self.image_namespace.queue = self.queue
        Thread(target=self.crawl_image).start()

    def crawl_image(self):
        self.socket.wait()

    def get_image_and_client(self, timeout=None):
        try:
            frame, client_id = self.queue.get(timeout=timeout)
            return frame, client_id
        except queue.Empty:
            return None, None

    def put_result(self, **args):
        client_id = args.get('client_id')
        status = args.get('status')
        message = {'clientSocketId': client_id}
        face_name = args.get('face_name')
        message['message'] = face_name
        message['status'] = status
        self.image_namespace.emit(Config.Socket.RESULT_EVENT, message)

    def release(self):
        # TODO: release this socket
        pass


class ImageNamespace(BaseNamespace):

    def connect(self, *args):
        print('Connect', args)

    def on_connect(self, *args):
        print('On connect', args)
        self.emit('join', os.environ.get('CV_SERVER_NAME', 'container-1234'))

    def on_image(self, *args):
        arg = args[0]
        url = arg.get('image', None)
        success, image = base64str_to_frame(url)
        if success:
            client_id = arg.get('clientSocketId', 'test-client')
            self.queue.put((image, client_id))


@catch_disconnect_all_class_methods
class ImageSocketDynamicPutResult(ImageSocket):

    def __init__(self, hostname, port, max_queue_size=50):
        imageio.plugins.freeimage.download()
        self.hostname = hostname
        self.port = port
        self.queue = multiprocessing.JoinableQueue(maxsize=max_queue_size)
        print('Connecting on %s:%s' %(self.hostname, self.port))
        self.socket = SocketIO(self.hostname, self.port)
        self.image_namespace = self.socket.define(AliasIdImageNamespace,
                                                  Config.Socket.IMAGE_NAMESPACE)
        self.image_namespace.set_queue(self.queue)
        # self.image_namespace.queue = self.queue
        Thread(target=self.crawl_image).start()

    def crawl_image(self):
        self.socket.wait()

    def get_image_and_client(self, timeout=None):
        try:
            # print(id(self.queue), 'get_image_and_client', self.queue.qsize())
            frame, client_id, alias_id = self.queue.get(timeout=timeout)
            self.queue.task_done()
            return frame, client_id, alias_id
        except queue.Empty:
            return None, None, None

    def put_result(self, **kwargs):
        self.image_namespace.emit(Config.Socket.RESULT_EVENT, kwargs)


class AliasIdImageNamespace(ImageNamespace):

    def set_queue(self, _queue):
        self.queue = _queue

    def on_image(self, *args):
        arg = args[0]
        url = arg.get('image', None)
        success, image = base64str_to_frame(url)
        if success:
            client_id = arg.get('clientSocketId', 'test-client')
            alias_id = arg.get('aliasId', 'test-aliasId')
            self.queue.put((image, client_id, alias_id))
            # if queue is full, wait until all items is processed before getting new one
            if self.queue.full():
                self.queue.join()
            # print(id(self.queue), 'AliasIdImageNamespace', self.queue.qsize())


@catch_disconnect_all_class_methods
class RetentionDashboardSocket():

    def __init__(self, hostname, port):
        imageio.plugins.freeimage.download()
        self.hostname = hostname
        self.port = port
        self.socket = SocketIO(self.hostname, self.port)
        self.retention_dashboard_namespace = self.socket.define(RetentionDashboardNamespace,
            Config.Socket.RETENTION_DASHBOARD_NAMESPACE)

    def send_result(self, **kwargs):
        image_id = kwargs.get('image_id')
        self.retention_dashboard_namespace.emit(Config.Socket.NEW_FACE_EVENT, image_id)


class RetentionDashboardNamespace(BaseNamespace):

    def on_connect(self, **args):
        print('Connect', **args)

@catch_disconnect_all_class_methods
class RetentionDashboardBlacklistSocket(RetentionDashboardSocket):

    def __init__(self, hostname, port):
        imageio.plugins.freeimage.download()
        self.hostname = hostname
        self.port = port
        self.update_blacklist = queue.Queue()
        self.socket = SocketIO(self.hostname, self.port)
        self.retention_dashboard_namespace = self.socket.define(RetentionDashboardBlacklistNamespace,
            Config.Socket.RETENTION_DASHBOARD_NAMESPACE)
        self.retention_dashboard_namespace.update_blacklist = self.update_blacklist
        self.socket.on(Config.Socket.UPDATE_BLACKLIST, \
                       self.retention_dashboard_namespace.on_update_pedestrian_blacklist())
        Thread(target=self.listen).start()

    def listen(self):
        self.socket.wait()

    def send_result(self, **kwargs):
        image_id = kwargs.get('image_id')
        self.retention_dashboard_namespace.emit(Config.Socket.NEW_FACE_EVENT, image_id)

    def get_update_blacklist(self):
        status = not self.update_blacklist.empty()
        if status:
            self.update_blacklist.get()
        return status

class RetentionDashboardBlacklistNamespace(RetentionDashboardNamespace):

    def on_connect(self, **args):
        print('Connect', **args)

    def on_update_pedestrian_blacklist(self, *args):
        print('Received event update blacklist')
        self.update_blacklist.put(0)

