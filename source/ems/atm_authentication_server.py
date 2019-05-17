import os
import time
import numpy as np
import multiprocessing
from collections import Counter
import cv2
import imageio
from bson.objectid import ObjectId
from ems import base_server
from pipe import pipeline, stage, task
from worker import face_detect_worker, face_preprocess_worker
from worker import face_extract_worker, atm_authentication_worker, face_verification_worker
from core import matcher, frame_reader
from core.cv_utils import base64str_to_frame, find_most_common
from utils import database, dict_and_list, socket_client
from utils import simple_request
from utils.logger import logger
from config import Config


class ATMAuthenticationServer(base_server.AbstractServer):

    def init(self):
        imageio.plugins.freeimage.download()
        self.database = database.ATMAuthenticationDatabase()
        self.matcher = matcher.KdTreeMatcher()
        self.matcher.build(self.database)
        self.recognition_pipeline = self.build_recognition_pipeline()
        self.manager = multiprocessing.Manager()
        self.result_dict = self.manager.dict()

        # for glasses mask
        volume_name = os.environ.get('VOLUME_NAME', '')
        self.path_prefix = os.path.join(volume_name, '_data')
        self.requester = simple_request.HTTPRequest(Config.MicroServices.SERVICE_GLASSES_MASK_CLASSIFICATION)

        try:
            self.socket_io = socket_client.ImageSocketDynamicPutResult(Config.Socket.HOST, Config.Socket.PORT, Config.Socket.MAX_QUEUE_SIZE)
            self.frame_reader = frame_reader.SocketIOAliasIdFrameReader(self.socket_io, timeout=Config.Server.FRAME_READER_TIMEOUT)
            print('Start multiprocess for recognition_socket_listener')
            p = multiprocessing.Process(target=self.recognition_socket_listener)
            p.start()
            pass
        except:
            print('Can not init socket stream')

    def add_endpoint(self):
        self.app.add_url_rule('/register', 'register', self.register_api, methods=['POST'])
        self.app.add_url_rule('/recognize', 'recognize', self.recognize_api, methods=['POST'])
        self.app.add_url_rule('/glasses-mask-classification', 'glasses-mask-classification', self.glasses_mask_classification_api, methods=['POST'])

    def recognition_socket_listener(self):
        images = []
        first_image_timestamp = 0
        while self.frame_reader.has_next():
            frame, client_id, alias_id = self.frame_reader.next_frame()
            #  if get a frame, usually the 1st one, start to record
            if frame is not None:
                if not images:
                    first_image_timestamp = time.time()
                    current_client_id = client_id
                images.append(frame)
                # if after time out and still not get any frame, we process frames a above
                # check if we still in this time session
                if (time.time() - first_image_timestamp) <= Config.Server.SOCKET_IMAGE_TIMEOUT:
                    if len(images) >= Config.Server.MAX_IMAGES_LENGTH:
                        input_data = {}
                        input_data['aliasId'] = alias_id
                        input_data['initTimestamp'] = time.time()
                        input_data['actionType'] = Config.ActionType.RECOGNIZE
                        input_data['client_id'] = time.time()

                        self.recognition(input_data, images)
                        data = self.get_client_result(input_data['client_id'])

                        status = data['status']
                        if status == Config.Status.SUCCESSFUL:
                            message = data.get('info_message', '')
                            personConfirmed = True
                        elif status == Config.Status.FAIL:
                            message = 'Face was not match'
                            personConfirmed = False
                        else:
                            status = Config.Status.FAIL
                            message = 'No face was found'
                            personConfirmed = False
                        response = {'status': status,
                                    'clientSocketId': current_client_id,
                                    'data': {'personConfirmed': personConfirmed},
                                    'message': message
                        }
                        print('from recognition_socket_listener', response)
                        self.socket_io.put_result(**response)
                        images = []
                else:
                    # after done processsing, clear the iamges cache
                    images = []
                    first_image_timestamp = 0

    def register_api(self):
        #get data from request
        input_data = self.validate_input_data()
        input_data['actionType'] = Config.ActionType.REGISTER

        images = self.input_images_parser()
        self.recognition(input_data, images)
        data = self.get_client_result(input_data['client_id'])

        print( 'from register_api', data)
        status = data.pop('status')
        if status == Config.Status.SUCCESSFUL:
            return self.response_success(data)
        elif status == Config.Status.FAIL:
            message = 'Previous registered faceId is: %s' % data.get('faceId', '')
        else:
            message = 'Faces not found'
        return self.response_error(message)

    def recognize_api(self):
        #get data from request
        input_data = self.validate_input_data()
        input_data['actionType'] = Config.ActionType.RECOGNIZE

        images = self.input_images_parser()
        self.recognition(input_data, images)
        data = self.get_client_result(input_data['client_id'])

        print('from recognize_api', data)
        status = data.pop('status')
        if status == Config.Status.SUCCESSFUL:
            message = data.pop('info_message', '')
            return self.response_success(data, message=message)
        elif status == Config.Status.FAIL:
            message = 'This person is not registered yet'
        else:
            message = 'Faces not found'
        return self.response_error(message)

    def recognition(self, input_data, images):
        # this client_id should be id to seperate between multi request that send to this same api

        print('\ngot {} images to {}'.format(len(images), input_data['actionType']))
        for image in images:
            _task = task.Task(task.Task.Frame)
            _task.package(frame=image, frame_info=input_data['client_id'])
            self.recognition_pipeline.put(_task)

        # notify pipeline there is no more images and start to matching
        _task = task.Task(task.Task.Event)
        _task.package(**input_data)
        self.recognition_pipeline.put(_task)

        data = self.recognition_pipeline.get()
        self.result_dict[input_data['client_id']] = data

    def glasses_mask_classification_api(self):
        '''
        This was create base on the assumption that the mask-glasses micro service
        is in the same physical server with this api, this might be a little confuse
        but it help speed up the file save/load io
        '''
        image_paths = self.save_and_get_images_path()
        relative_paths = [os.path.join(self.path_prefix, path) for path in image_paths]
        predictions = self.requester.post_list(Config.MicroServices.IMAGES, relative_paths)
        if predictions is not None and ('predictions' in predictions):
            result_list = np.array(predictions['predictions'])
            has_glasses = result_list[:, 0]
            has_mask = result_list[:, 1]
            if has_glasses.any() and has_mask.any():
                has_glasses, _ = Counter(list(has_glasses)).most_common(1)[0]
                has_mask, _ = Counter(list(has_mask)).most_common(1)[0]

        # currently we don't save these record to database
            if (has_glasses is not None) and (has_mask is not None):
                data = {'has_glasses': has_glasses,
                        'has_mask': has_mask}
                return self.response_success(data)
        message = 'Process error, can not locate faces'
        return self.response_error(message)

    def build_recognition_pipeline(self):
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_verification_worker.PreprocessDetectedFaceWorker, 1)
        stageCollect = stage.Stage(atm_authentication_worker.ATMAuthenticationCollectWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.ArcFacesEmbeddingExtractWorker, 1)
        stageMatching = stage.Stage(atm_authentication_worker.ATMAuthenticationMatchingWorker,
                                size=1,
                                database=self.database,
                                matcher=self.matcher)
        stageDatabase = stage.Stage(atm_authentication_worker.ATMAuthenticationDatabaseWorker, 1,
                                   database=self.database)

        stageDetectFace.link(stagePreprocess)
        stagePreprocess.link(stageCollect)
        stageCollect.link(stageExtract)
        stageExtract.link(stageMatching)
        stageMatching.link(stageDatabase)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline

    def get_client_result(self, client_id):
        data = self.result_dict.pop(client_id, None)
        while data is None:
            data = self.result_dict.pop(client_id, None)
        return data

    def validate_input_data(self):
        # getting data
        input_data = {}
        input_data['aliasId'] = self.request.form.get('aliasId')
        input_data['initTimestamp'] = float(self.request.form.get('initTimestamp', -1))
        input_data['receiveTimestamp'] = round(time.time(), 3)
        input_data['client_id'] = time.time()

        alias_id = input_data['aliasId']
        if (alias_id is not None) and not (self.database.is_existed_aliasID(alias_id)):
            message = 'aliasID {} is not existed in database'.format(alias_id)
            print(message)
            return self.response_error(message)
        return input_data

    def input_images_parser(self):
        # parsing images
        images = []
        if 'video' in self.request.files:
            save_dir = str(time.time())
            self.request.files.get('video').save(save_dir)
            cap = cv2.VideoCapture(save_dir)
            ret, frame = cap.read()
            while ret:
                images.append(frame)
                ret, frame = cap.read()
            os.remove(save_dir)
        elif 'images[]' in self.request.form:
            images_str = self.request.form.getlist('images[]')
            for image_str in images_str:
                success, frame = base64str_to_frame(image_str)
                if success:
                    images.append(frame)
        # convert image channel
        images_cvted = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        return images_cvted

    def save_and_get_images_path(self):
        image_paths = []
        if 'video' in self.request.files:
            save_dir = str(time.time())
            self.request.files.get('video').save(save_dir)
            cap = cv2.VideoCapture(save_dir)
            ret, frame = cap.read()
            while ret:
                image_name = '%s.jpg' % time.time()
                image_path = os.path.join(Config.Dir.DATA_DIR, image_name)
                cv2.imwrite(image_path, frame)
                image_paths.append(image_name)
                ret, frame = cap.read()
        elif 'images[]' in self.request.form:
            images_str = self.request.form.getlist('images[]')
            for image_str in images_str:
                success, frame = base64str_to_frame(image_str)
                if success:
                    image_name = '%s.jpg' % time.time()
                    image_path = os.path.join(Config.Dir.DATA_DIR, image_name)
                    cv2.imwrite(image_path, frame)
                    image_paths.append(image_name)
        return image_paths
