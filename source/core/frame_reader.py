"""
This script contains all handlers for reading image frames from different sources
"""
from config import Config
import os
import glob
from abc import ABCMeta, abstractmethod
import cv2
from scipy import misc
from queue import Queue
import queue
import time
from collections import defaultdict
import requests
from utils import timestamp_getter

class AbstractFrameReader(metaclass=ABCMeta):
    '''
    Abstract class to provide frame
    '''
    def __init__(self, scale_factor=1, should_crop=False, re_source = False, \
                                                        timeout = Config.Frame.STREAM_TIMEOUT):
        '''
        :param scale_factor: reduce frame factor
        '''
        self._scale_factor = scale_factor
        self._should_crop = should_crop
        self.re_source = re_source
        self.timeout = timeout

    @abstractmethod
    def next_frame(self):
        '''
        Return next frame, None if empty
        '''
        pass

    def has_next(self):
        '''
        Check if has more frame to read
        '''
        return True

    def _scale_frame(self, frame):
        '''
        Rescale frame for faster processing
        :param frame: input frame to rescale
        :return frame: resized frame
        '''
        if frame is None:
            return None

        if self._scale_factor > 1:
            frame = cv2.resize(frame, (int(len(frame[0]) / self._scale_factor),
                                       int(len(frame) / self._scale_factor)))
        return frame

    def _crop_roi(self, frame):
        '''
        Crop roi for faster downstream processing
        :param frame: input frame to crop
        :return frame: cropped frame
        '''
        if frame is None:
            return None

        if self._should_crop:
            roi = AbstractFrameReader._roi(frame, self._scale_factor)
            frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
        return frame

    @staticmethod
    def _roi(frame, scale):
        '''
        Find ROI of interest for particular frame
        '''
        # video_out_fps, video_out_w, video_out_h = self.get_info()
        # self.roi = [int(center[0]-Config.Frame.ROI_CROP[0]*video_out_w),
        #             int(center[1]-video_out_h*Config.Frame.ROI_CROP[1]),
        #             int(center[0]+Config.Frame.ROI_CROP[2]*video_out_w),
        #             int(center[1]+Config.Frame.ROI_CROP[3]*video_out_h)]
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        roi = [
            int(center[0] - Config.Frame.ROI_CROP[0] * w),
            int(center[1] - Config.Frame.ROI_CROP[1] * h),
            int(center[0] + Config.Frame.ROI_CROP[2] * w),
            int(center[1] + Config.Frame.ROI_CROP[3] * h)
        ]
        return roi

    # @abstractmethod
    def release(self):
        '''
        Release this frame reader
        '''
        pass

    def get_info(self):
        '''
        Get current frame reader info
        '''
        pass


class RabbitFrameReader(AbstractFrameReader):
    '''
    Read frame from rabbit queue, for register for live image
    '''

    def __init__(self,
                 rb,
                 queue_name,
                 scale_factor=Config.Frame.SCALE_FACCTOR,
                 should_crop=False):
        '''
        :param rb: rabbitmq instance to read new frame from
        :param scale_factor: reduce frame factor
        :param should_crop: should crop roi
        '''
        self.__rb = rb
        self.__queue_name = queue_name
        if self.__rb.is_exist(self.__queue_name):
            raise ValueError("Queue not exist")
        super(RabbitFrameReader, self).__init__(scale_factor, should_crop)

    def next_frame(self):
        '''
        Read next frame from rabbit, may return None if there is no frame avaiable
        '''
        frame = self.__rb.receive_raw_live_image(self.__queue_name)
        frame = self._crop_roi(frame)
        frame = self._scale_frame(frame)
        return frame

    def release(self):
        pass

    def get_info(self):
        '''
        Get information, omit one frame to get info
        '''
        frame = None
        while frame is None:
            frame = self.next_frame()
        return 24, frame.shape[1], frame.shape[0]


class URLFrameReader(AbstractFrameReader):
    """
    Read frame from video stream or from video path or web cam
    """

    # TODO: Should create 3 subclass?
    WEBCAM = 0
    VIDEO_FILE = 1
    IP_STREAM = 2

    def __init__(self, cam_url, scale_factor=Config.Frame.SCALE_FACCTOR, re_source = False,
                    timeout = Config.Frame.STREAM_TIMEOUT, should_crop=False, convert_color=True):
        '''
        :param cam_url: url for video stream
        :param scale_factor: reduce frame factor
        '''
        self.cam_url = cam_url
        if type(cam_url) == int or cam_url.isdigit():
            cam_url = int(cam_url)
            self.__url_type = URLFrameReader.WEBCAM
        elif self.__is_file(cam_url):
            self.__url_type = URLFrameReader.VIDEO_FILE
        else:
            self.__url_type = URLFrameReader.IP_STREAM
        self.__video_capture = cv2.VideoCapture(cam_url)
        self.stream_time = time.time()
        self.convert_color = convert_color
        super(URLFrameReader, self).__init__(scale_factor, should_crop,
                                             re_source, timeout)

    def next_frame(self):
        '''
        Return scaled frame from video stream
        '''
        _, frame = self.__video_capture.read()
        if frame is not None:
            if self.convert_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._crop_roi(frame)
            frame = self._scale_frame(frame)
            self.stream_time = time.time()
        else:
            if self.re_source and time.time() - self.stream_time > self.timeout:
                print("reconnect to stream")
                self.__init__(self.cam_url, self._scale_factor, self.re_source,
                              self.timeout, self._should_crop)

        return frame

    def has_next(self):
        '''
        If the video stream is limited (like video file),
            the frame will return False when it's read all the frame
        else (ip stream or webcam) it will always return True
        '''
        if self.__url_type == URLFrameReader.VIDEO_FILE:
            return self.__video_capture.get(cv2.CAP_PROP_POS_FRAMES) \
                    < self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            return self.__video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) != 0 \
                   and self.__video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) != 0

    def release(self):
        '''
        Release the video source
        '''
        self.__video_capture.release()

    def __is_file(self, cam_url):
        '''
        Check if cam_url is a video in filesystem or an ip stream
        :param cam_url: url to check
        :return True if cam_url exist in file system
        '''
        return os.path.exists(cam_url)

    def get_info(self):
        '''
        Frame is now depend also on ROI, not just simple scale factor
        Get information, omit one frame to get info
        '''
        frame = None
        while frame is None:
            frame = self.next_frame()

        return self.__video_capture.get(cv2.CAP_PROP_FPS), \
            frame.shape[1], frame.shape[0]


class DirectoryFrameReader(AbstractFrameReader):
    """
    Read all image from a directory

    >>> frame_reader = DirectoryFrameReader(r'../data/matching/set1', 'jpg'); \
        len(frame_reader._DirectoryFrameReader__image_files)
    9
    >>> for i in range(9):
    ...     _ = frame_reader.next_frame()
    >>> frame_reader.has_next()
    False
    """

    def __init__(self,
                 dir_,
                 ext='jpg',
                 scale_factor=Config.Frame.SCALE_FACCTOR,
                 should_crop=False):
        '''
        :param dir_: directory that contains images
        :pram ext: image extension to read
        :param scale_factor: reduce frame factor
        '''
        self.__image_files = glob.glob(os.path.join(dir_, '*.%s' % ext))
        self.__image_files.sort()
        self.__frame_index = 0
        super(DirectoryFrameReader, self).__init__(scale_factor, should_crop)

    def next_frame(self):
        '''
        Read next image from directory
        '''
        frame = cv2.imread(self.__image_files[self.__frame_index])
        frame = self._scale_frame(frame)
        frame = self._crop_roi(frame)
        self.__frame_index += 1
        return frame

    def has_next(self):
        '''
        Return False when all images have been read
        '''
        return self.__frame_index < len(self.__image_files)

    def release(self):
        '''
        Release all image file
        '''
        self.__frame_index = 0
        self.__image_files = []


class QueueFrameReader(AbstractFrameReader):
    """
    Read all image from a queue
    """

    def __init__(self, scale_factor=Config.Frame.SCALE_FACCTOR):
        '''
        :param dir_: directory that contains images
        :pram ext: image extension to read
        :param scale_factor: reduce frame factor
        '''
        super(QueueFrameReader, self).__init__(scale_factor)
        self.queue = Queue()

    def next_frame(self):
        '''
        Read next image from directory
        '''
        try:
            frame = self.queue.get(False)
        except queue.Empty:
            frame = None
        return frame

    def has_next(self):
        '''
        Return False when all images have been read
        '''
        return not self.queue.empty()

    def release(self):
        '''
        Release all image file
        '''
        pass

    def add_item(self, item):
        self.queue.put(item)

    def clear(self):
        self.queue = Queue()


class SocketIOFrameReader(AbstractFrameReader):
    """
    Read frame from socket io stream
    """

    def __init__(self, image_socket, timeout=10):
        '''
        '''
        self.image_socket = image_socket
        self.timeout = timeout
        super(SocketIOFrameReader, self).__init__()

    def next_frame(self):
        '''
        Return read frame from socketio stream
        '''
        frame, client_id = self.image_socket.get_image_and_client(self.timeout)
        return frame, client_id

    def has_next(self):
        # always waiting for new image
        return True

    def release(self):
        self.image_socket.release()


class SocketIOAliasIdFrameReader(SocketIOFrameReader):
    def next_frame(self):
        '''
        Return read frame from socketio stream
        '''
        frame, client_id, alias_id = self.image_socket.get_image_and_client(self.timeout)
        return frame, client_id, alias_id


class NASFrameReader(AbstractFrameReader):
    '''
        Read Frame from NAS with chosen FPS
    '''

    def __init__(self, _database, **kwargs):
        self.database = _database
        self.domain = kwargs.get('domain')
        self.account = kwargs.get('account')
        self.password = kwargs.get('pass')
        self.version = kwargs.get('version')
        self.video_duration = kwargs.get('video_duration',5)
        self.number_of_reserved_video = kwargs.get('number_of_reserved_video', 2)
        if self.version == None:
            print('No version specified, use default version of NAS')
            self.version = 6
        self.cameraID = kwargs.get('cameraID')
        if self.cameraID == None:
            print('No cameraID specified, use camera number 1')
            self.cameraID = 1
        self.wanted_fps = kwargs.get('wantedFPS')
        if self.wanted_fps == None:
            print('No FPS specified, use default FPS = 6 ')
            self.wanted_fps = 6
        self.last_time_stamps = kwargs.get('fromTime')
        if self.last_time_stamps == None:
            print('No timestamp specified, Find from database')
        elif self.last_time_stamps == -1:
            print('No timestamp specified, Process Real Time')
        self.timestamp_getter = timestamp_getter.TimestampGetter()
        self.set_up(self.last_time_stamps)
        super(NASFrameReader, self).__init__()

    def set_up(self, from_time):
        self.sid = self.authenticate()
        print('Init NASFrameReader')
        if from_time == -1:
            self.last_time_stamps = int(time.time() - self.number_of_reserved_video*60*self.video_duration)
            print('[SETUP] Real Time Set Up')
        elif from_time == None:
            self.last_time_stamps = self.database.find_time_stamp()
            print('[SETUP] Play from timestamp : ', self.last_time_stamps)
        self.take_record_list(cameraID=self.cameraID, version=self.version, fromtime=self.last_time_stamps)


    def authenticate(self):
        try:
            ck = requests.get('{}/auth.cgi?api=SYNO.API.Auth&account={}&passwd={}&version={}&method=Login&format=sid'.format(self.domain, self.account, self.password, self.version))
            if ck.status_code == 200:
                print('[AUTH] Authenication success')
                sid = ck.json()['data']['sid']
                return sid
            else :
                print('[AUTH] Authenication failed, status code = ',ck.status_code)
                return None
        except:
            return None
        

    def take_record_list(self, cameraID, fromtime, version = 6):
        success = False
        data = None
        while not success:
            try:
                ck = requests.get('{}/entry.cgi?api=SYNO.SurveillanceStation.Recording&cameraIds={}&version={}\
                                &method=List&fromTime={}&_sid={}'.format(self.domain, cameraID, version, fromtime, self.sid))
                if ck.status_code == 200 :
                    if 'error' in ck.json():
                        self.sid = self.authenticate()
                        print('[AUTH] Successfully ReAuthenticate')
                    else:
                        try: 
                            data = ck.json()['data']
                            success = True
                        except:
                            continue
                else:
                    print('[LIST] Failed to take record list, status code = ',ck.status_code)
            except KeyboardInterrupt:
                break
            except :
                print('CANNOT CONNECT TO RECORD STORAGE')
                self.sid = self.authenticate()
                if self.sid is None:
                    print('[AUTH] Unable to ReAuthenticate')
                else:
                    print('[Reconnecting] Taking record list')
        if success == True:
            self.next_ID = None
            self.last_ID = None
            self.count = 0
            self.array_count = 0
            if data is None:
                data = ck.json()['data']
            self.array_of_indexes = []
            if data['total'] > 1:
                self.next_time_stamps = self.timestamp_getter.nas_timestamp(data['recordings'][-2]['filePath'])
                self.last_time_stamps = self.timestamp_getter.nas_timestamp(data['recordings'][-1]['filePath'])
                self.next_ID = data['recordings'][-2]['id']
                self.last_ID = data['recordings'][-1]['id']
                self.cameraName = data['recordings'][-1]['cameraName']
                print('[INFO] {}'.format(data['recordings'][-1]))
            elif data['total'] == 1 :
                self.next_time_stamps = None
                self.last_time_stamps = self.timestamp_getter.nas_timestamp(data['recordings'][-1]['filePath'])
                self.last_ID = data['recordings'][-1]['id']
                self.cameraName = data['recordings'][-1]['cameraName']
            else:
                self.next_time_stamps = None
                self.cameraName = 'NoMoreVideo'
            if not self.database.is_exist_timestamp(recordTimestamp=self.last_time_stamps,
                                                    isDone=False,
                                                    cameraName=self.cameraName):
                self.database.insert_new_timestamp(recordTimestamp=self.last_time_stamps,
                                                    isDone=False,
                                                    cameraName=self.cameraName)
            self.start_time = time.time()
            print('Changing to record ID :', self.last_ID)
            print('Next ID :', self.next_ID)
            if self.last_ID is not None:
                self.video_url = '{}/entry.cgi?api=SYNO.SurveillanceStation.Recording&recordingId={}&version={}\
                                    &method=Stream&videoCodec=1&_sid={}'.format(self.domain, self.last_ID, version,self.sid)
                self.__video_capture = cv2.VideoCapture(self.video_url)
                self.__total_frames = self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                self._fps = self.__video_capture.get(cv2.CAP_PROP_FPS)
                self.interval = (self._fps/(self.wanted_fps ))
                if self.interval < 1:
                    self.interval = 1
                # print(self.interval)
                for i in range(int(self.__total_frames/self.interval)):
                    self.array_of_indexes.append( int(i*self.interval) )
                print(' taking {} frame out of {}'.format(len(self.array_of_indexes),
                                                        self.__total_frames))


    def next_frame(self):
        if (self.next_ID == None ):
            print('Checking video in storage')
            self.take_record_list(cameraID=self.cameraID,
                                    fromtime=self.last_time_stamps)
            if self.next_ID == None:
                if self.last_ID == None:
                    print('No video in storage')
                else:
                    print('Waiting for finished record video')
                print('Sleeping {} min to wait for next video'.format(self.video_duration))
                time.sleep(self.video_duration*60)
                return None, 0
        frame = None
        if self.array_count < len(self.array_of_indexes):
            while(self.count <= self.array_of_indexes[self.array_count]):
                ret, frame = self.__video_capture.read()
                self.count += 1
                flag = self.count > self.array_of_indexes[self.array_count]
                if flag :
                    self.array_count += flag
                    break
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._crop_roi(frame)
            frame = self._scale_frame(frame)
            return frame, time.time()-self.start_time + self.last_time_stamps
        else:
            print('total processing time of video {} : {}'.format(self.last_ID,
                                                           time.time()-self.start_time))
            if self.authenticate() is not None:
                self.database.update_done(recordTimestamp=self.last_time_stamps)
                self.last_time_stamps = self.last_time_stamps + 1
                self.take_record_list(self.cameraID, fromtime=self.last_time_stamps)
            else:
                print('Reconnecting .....')
                self.take_record_list(self.cameraID, fromtime=self.last_time_stamps)
            return None, 0


    def has_next(self):
        print( ' So sanh : {} va {}'.format(self.__video_capture.get(cv2.CAP_PROP_POS_FRAMES), self._total_frames))
        return (self.__video_capture.get(cv2.CAP_PROP_POS_FRAMES) < self._total_frames)


    def release(self):
        self.__video_capture.release()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
