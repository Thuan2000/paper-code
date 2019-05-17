import numpy as np
from pipe import worker, task
from pipe.trace_back import process_traceback
from utils import dict_and_list
from core.cv_utils import find_most_common
from config import Config


class EcommerceCollectWorker(worker.Worker):

    def doInit(self):
        self.last_frame_with_face = 0
        max_nrof_faces = 3
        self.waiting_images = dict_and_list.WaitingImageQueue(max_nrof_faces)
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        faces, preprocessed_images, frame, frame_info = \
                data['faces'], data['images'], data['frame'], data['frame_info']
        client_id = frame_info

        # demo, only add first face
        self.waiting_images.put(client_id, preprocessed_images[0])
        print('=' * 10, 'Process image', client_id)
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


class EcommerceMatchingWorker(worker.Worker):

    def __init__(self, **args):
        self.matcher = args.get('matcher')
        self.database = args.get('database')
        self.embs = None

    def doFaceTask(self, _task):
        data = _task.depackage()
        new_embs, frame_info = data['embs'], data['frame_info']
        if self.embs is None:
            self.embs = new_embs
        else:
            self.embs = np.vstack((self.embs, new_embs))

    def doFrameTask(self, _task):
        # Omit _task
        pass

    def doEventTask(self, _task):
        '''
        An event _task indicate that we have feed all images from api request
        and ready to return result
        '''
        print('=' * 10, 'ecommerce matching')
        top_ids, dists = self.matcher.match_arr(self.embs, return_dists=True)
        predict_ids = []
        for i in range(len(top_ids)):
            predict_id = find_most_common(top_ids[i])
            min_dist = min(dists[i])
            if min_dist > 0.65:
                predict_id = Config.Matcher.NEW_FACE
            predict_ids.append(predict_id)
        print('predict_ids', set(predict_ids))
        predict_id = find_most_common(predict_ids)
        if predict_id == Config.Matcher.NEW_FACE:
            print('New face')
            #TODO: Refactor database
            _id = self.database.mongodb_face.insert({
                'embeddings':
                self.embs.tolist()
            })
            new_face_id = str(_id)
            print(new_face_id)
            labels = np.array([new_face_id] * len(self.embs))
            self.matcher.update(self.embs, labels)
            self.embs = None
            self.putResult(new_face_id)
        else:
            print('Match face', predict_id)
            self.putResult(predict_id)
