import numpy as np
from pipe import worker, task
from utils import dict_and_list
from core import preprocess
from core.cv_utils import CropperUtils, compute_nearest_neighbors_distances
from config import Config


class PreprocessDetectedFaceWorker(worker.Worker):

    def doInit(self):
        self.preprocessor = preprocess.Preprocessor()
        print(self.name, '=' * 10)

    def doFaceTask(self, _task):
        data = _task.depackage()
        bbs, pts, frame, frame_info = data['bbs'], data['pts'], data[
            'frame'], data['frame_info']
        bbox = self.get_biggest_face(bbs)
        if bbox.any():
            cropped_face = CropperUtils.crop_face(frame, bbox[:-1])
            preprocessed = self.preprocessor.process(cropped_face)

            _task = task.Task(task.Task.Face)
            _task.package(images=preprocessed, frame_info=frame_info)
            self.putResult(_task)

    def get_biggest_face(self, bboxes):

        def area(bbox):
            return (bbox[3]-bbox[1])*(bbox[2]-bbox[0])

        areas = [area(bbox) for bbox in bboxes]
        if areas:
            max_area = max(areas)
            max_area_idx = areas.index(max_area)
            biggest_bbox = bboxes[max_area_idx]
            return biggest_bbox
        return np.array([])


class CollectWorker(worker.Worker):

    def doInit(self):
        self.waiting_images = dict_and_list.WaitingImageQueue(max_size=3)
        print(self.name, '='*10)

    def doFaceTask(self, _task):
        data = _task.depackage()
        preprocessed_images, frame_info = data['images'], data['frame_info']
        client_id = frame_info
        self.waiting_images.put(client_id, preprocessed_images)
        print('='*10, 'Process image', client_id)
        if self.waiting_images.has_enough(client_id):
            images = self.waiting_images.get(client_id)
            images = np.array(images)
            _task = task.Task(task.Task.Face)
            _task.package(images=images, frame_info=frame_info)
            self.putResult(_task)

    def doEventTask(self, _task):
        waiting_images = self.waiting_images.get_all()
        for client_id, images in waiting_images:
            images = np.array(images)
            _face_task = task.Task(task.Task.Face)
            _face_task.package(images=images, frame_info=client_id)
            self.putResult(_face_task)
        self.putResult(_task)


class MatchingWorker(worker.Worker):

    def doInit(self):
        self.embs_dict = dict_and_list.EmbsDict(max_size=float('Inf'))

    def doFaceTask(self, _task):
        data = _task.depackage()
        embs, frame_info = data['embs'], data['frame_info']
        if embs.any():
            self.embs_dict.put(frame_info, embs)

    def doEventTask(self, _task):
        data = _task.depackage()
        client_id_1, client_id_2 = data['frame_info_1'], data['frame_info_2']
        embs_1 = self.embs_dict.pop(client_id_1)
        embs_2 = self.embs_dict.pop(client_id_2)

        if embs_1.size == 0 or embs_2.size == 0:
            data = {'status': Config.Status.NO_FACES,
                    'message': 'Faces not found'}
            self.putResult(data)
            return
        # choose embs with more values as matcher
        if embs_1.size > embs_2.size:
            matcher_embs = embs_1
            source_embs = embs_2
        else:
            matcher_embs = embs_2
            source_embs = embs_1
        nrof_test_samples = source_embs.shape[0]
        distances = compute_nearest_neighbors_distances(matcher_embs, source_embs)
        print(distances)
        print('average distance:', np.mean(distances))
        thresholding = distances < Config.Matching.MATCHING_THRESHOLD
        passed_thresholing = thresholding.sum() / nrof_test_samples
        if passed_thresholing > Config.Matching.POPULARITY:
            is_same_person = True
        else:
            is_same_person = False
        data = {'status': Config.Status.SUCCESSFUL,
                'isSamePerson': is_same_person,
                'confidentScore': passed_thresholing}
        self.putResult(data)
