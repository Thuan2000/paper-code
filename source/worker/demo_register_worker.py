import time
import numpy as np
from pipe import worker, task
from pipe.trace_back import process_traceback
from utils.logger import logger
from utils import dict_and_list
from core.cv_utils import find_most_common
from config import Config


class RegisterCollectWorker(worker.Worker):

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


class RegisterMatchingWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')
        self.matcher = args.get('matcher')
        self.socket = args.get('socket')
        max_hold_embs = 12
        self.embs_dict = dict_and_list.EmbsDict(max_hold_embs)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        embs, frame_info = data['embs'], data['frame_info']

        client_id = frame_info
        self.embs_dict.put(client_id, embs)
        matching_emb = self.embs_dict.get(client_id)

        top_ids, dists = self.matcher.match_arr(matching_emb, return_dists=True)
        predict_ids = []
        for i in range(len(top_ids)):
            predict_id = find_most_common(top_ids[i])
            min_dist = min(dists[i])
            if min_dist > 0.65:
                predict_id = Config.Matcher.NEW_FACE
            predict_ids.append(predict_id)
        print('=' * 10, 'Match faces', predict_ids)
        predict_id = find_most_common(predict_ids)
        self.socket.put_result(status='successful', client_id=client_id, face_name=predict_id)

    @process_traceback
    def doFrameTask(self, _task):
        """
        No face detected for this frame, but still sent result back to
        """
        data = _task.depackage()
        frame_info = data['frame_info']
        client_id = frame_info
        self.socket.put_result(status='alert', client_id=client_id, face_name='')

    @process_traceback
    def doEventTask(self, _task):
        data = _task.depackage()
        face, client_id, send_at = data['face'], data['client_id'], data['sent_at']
        embs = self.embs_dict.get(client_id)
        # check if embs is available
        if embs is not None:
            labels = [face['name']] * len(embs)
            self.matcher.update(embs, labels)
            self.database.insert_new_face(face=face, embs=embs)
