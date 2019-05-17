import time
import numpy as np
from pipe import worker, task
from pipe.trace_back import process_traceback
from utils import dict_and_list
from core.cv_utils import find_most_common, compute_nearest_neighbors_distances
from config import Config


class ATMAuthenticationCollectWorker(worker.Worker):

    def doInit(self):
        self.last_frame_with_face = 0
        max_nrof_faces = 3
        self.waiting_images =  dict_and_list.WaitingImageQueue(max_nrof_faces)
        print(self.name, '='*10)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        preprocessed_images, frame_info = data['images'], data['frame_info']
        client_id = frame_info

        # demo, only add first face
        if preprocessed_images.any():
            self.waiting_images.put(client_id, preprocessed_images)
        # print('='*10, 'Process image from client_id', client_id)
        if self.waiting_images.has_enough(client_id):
            images = self.waiting_images.get(client_id)
            images = np.array(images)
            _task = task.Task(task.Task.Face)
            _task.package(images=images, frame_info=frame_info)
            self.putResult(_task)

    @process_traceback
    def doEventTask(self, _task):
        # handle remaining images
        data = _task.depackage()
        client_id = data['client_id']
        images = self.waiting_images.get(client_id)
        images = np.array(images)
        face_task = task.Task(task.Task.Face)
        face_task.package(images=images, frame_info=client_id)
        self.putResult(face_task)
        # notify that no more images by delegate event
        self.putResult(_task)


class ATMAuthenticationMatchingWorker(worker.Worker):
    '''
    Use aliasId as lables for matcher
    faceId will be ObjectId query in database
    '''
    def __init__(self, **args):
        self.matcher = args.get('matcher')
        self.database = args.get('database')
        self.embs_list = dict_and_list.EmbsDict(float('Inf'))

    def doFaceTask(self, _task):
        data = _task.depackage()
        new_embs, frame_info = data['embs'], data['frame_info']
        if new_embs.any():
            self.embs_list.put(frame_info, new_embs)

    def doEventTask(self, _task):
        '''
        An event _task indicate that we have feed all images from api request
        and ready to return result
        '''
        print('='*10, 'ATMAuthentication matching')
        data = _task.depackage()
        client_id = data['client_id']
        embs = self.embs_list.get(client_id)
        if (embs is None) or (embs.size == 0):
            data['status'] = Config.Status.NO_FACES
            _task = task.Task(task.Task.Event)
            _task.package(**data)
            self.putResult(_task)
            return

        if data['actionType'] == Config.ActionType.REGISTER:
            self.handleRegister(embs, data)
        elif data['actionType'] == Config.ActionType.RECOGNIZE:
            self.handleRecognize(embs, data)

    def handleRegister(self, embs, data):
        predict_id = self.recognize(embs)
        if predict_id == Config.Matcher.NEW_FACE:
            if data['aliasId'] is None:
                data['aliasId'] = str(time.time())
            labels = np.array([data['aliasId']] * len(embs))
            self.matcher.update(embs, labels)
            data['status'] = Config.Status.SUCCESSFUL
            data['embeddings'] = embs
            _task = task.Task(task.Task.Face)
        else:
            data['aliasId'] = predict_id
            data['status'] = Config.Status.FAIL
            _task = task.Task(task.Task.Event)

        _task.package(**data)
        self.putResult(_task)

    def handleRecognize(self, new_embs, data):
        data['status'] = Config.Status.FAIL
        if data['aliasId'] is not None:
            nrof_test_samples = new_embs.shape[0]
            source_embs = self.database.get_embs_by_face(data['aliasId'])
            if source_embs.size != 0:
                distances = compute_nearest_neighbors_distances(new_embs, source_embs)
                thresholding = distances < Config.Matching.MATCHING_THRESHOLD
                passed_thresholing = thresholding.sum() / nrof_test_samples
                print('client_id {}, matching passed result: {}'.format(data['client_id'], passed_thresholing))
                if passed_thresholing > Config.Matching.POPULARITY:
                    data['status'] = Config.Status.SUCCESSFUL
        else:
            predict_id = self.recognize(new_embs)
            if predict_id != Config.Matcher.NEW_FACE:
                data['aliasId'] = predict_id
                data['status'] = Config.Status.SUCCESSFUL

        _task = task.Task(task.Task.Event)
        _task.package(**data)
        self.putResult(_task)

    def recognize(self, embs):
        top_ids, dists = self.matcher.match_arr(embs, return_dists=True)
        predict_ids = []
        for i in range(len(top_ids)):
            predict_id = find_most_common(top_ids[i])
            min_dist = min(dists[i])
            if min_dist > Config.Matching.MATCHING_THRESHOLD:
                predict_id = Config.Matcher.NEW_FACE
            predict_ids.append(predict_id)
        print('predict_ids:', set(predict_ids))
        predict_id = find_most_common(predict_ids)
        print('predict id: %s, popularity: %s' % (predict_id, predict_ids.count(predict_id)/len(predict_ids)))
        return predict_id


class ATMAuthenticationDatabaseWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')

    # only use by register mode
    def doFaceTask(self, _task):
        data = _task.depackage()
        embs = data.pop('embeddings')
        aliasId = data['aliasId']
        _id = self.database.mongodb_face.insert({'embeddings': embs.tolist(),
                                                 'aliasId': aliasId})
        face_id = str(_id)
        data['faceId'] = face_id
        self.insertStatistic(data)

    def doEventTask(self, _task):
        print('ATMAuthenticationDatabaseWorker')
        data = _task.depackage()
        self.insertStatistic(data)

    def insertStatistic(self, data):
        if 'aliasId' in data:
            faceId = self.database.get_faceId_by_aliasID(data['aliasId'])
            data['faceId'] = faceId
            user_infos = self.database.get_user_info(data['aliasId'])
            if user_infos is not None:
                name, phone, email = user_infos
                data['info_message'] = 'Hello %s, phone: %s, email: %s' % (name, phone, email)
        data['endProcessTimestamp'] = round(time.time(), 3)
        self.database.insert_statistic(**data)
        return_keys = ['status', 'faceId', 'aliasId', 'info_message']
        data = {k:v for k,v in data.items() if ((k in return_keys) and (v is not None))}
        self.putResult(data)
