from statistics import mode
from pipe import worker, task
from pipe.trace_back import process_traceback
from core import preprocess, mask_glasses
from utils import dict_and_list
import numpy as np

# TODO: This file may not needed in the future, consider delete it


class PreprocessFrameWorker(worker.Worker):
    '''
    This worker do normalize the frame before doing classification classifier
    '''

    def doInit(self):
        self.preprossor = preprocess.Preprocessor(algs=preprocess.normalization)

    @process_traceback
    def doFaceTask(self, _task):
        '''
        Only process frame with face
        '''
        data = _task.depackage()
        frame, frame_info = data['frame'], data['frame_info']
        preprocessed_frame = self.preprossor.process(frame)
        _task = task.Task(task.Task.Face)
        _task.package(frame=preprocessed_frame, frame_info=frame_info)
        self.putResult(_task)


class GlassesMaskExtractWorker(worker.Worker):

    def doInit(self):
        self.glasses_classifier = mask_glasses.GlassesClassifier()
        self.mask_classifier = mask_glasses.MaskClassifier()
        self.waiting_images = dict_and_list.WaitingImageQueue(max_size=3)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        frame, frame_info = data['frame'], data['frame_info']
        client_id = frame_info

        self.waiting_images.put(client_id, frame)
        if self.waiting_images.has_enough(client_id):
            images = self.waiting_images.get(client_id)
            images = np.array(images)
            has_masks = self.mask_classifier.is_wearing_mask(images)
            has_glasses = self.glasses_classifier.is_wearing_glasses(images)
            _task = task.Task(task.Task.Face)
            _task.package(
                glasses=has_glasses, masks=has_masks, client_id=client_id)
            self.putResult(_task)


class GlassesMaskMatchingWorker(worker.Worker):

    def __init__(self, **args):
        self.socket = args.get('socket')
        self.result_list = dict_and_list.EmbsDict(max_size=6)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        has_glasses, has_masks = data['glasses'], data['masks']
        client_id = data['client_id']
        results = np.array([has_glasses, has_masks])
        self.result_list.put(client_id, results.T)

        matching_results = self.result_list.get(client_id)
        print('Matching result', matching_results.shape)
        has_glasses = GlassesMaskMatchingWorker.find_most_common(
            matching_results[:, 0])
        has_mask = GlassesMaskMatchingWorker.find_most_common(
            matching_results[:, 1])
        result = 'Has glasses %s, has mask %s' % (has_glasses, has_mask)
        print(result)
        # self.image_socket.put_result(client_id=client_id, status='successful', face_name=result)

    @process_traceback
    def doFrameTask(self, _task):
        print('No face detected')
        # self.image_socket.put_result(client_id=client_id, status='alert', face_name='Detecting')

    @staticmethod
    def find_most_common(ls):
        # TODO: Move to utils
        try:
            top = mode(ls)
        except:
            top = ls[0]
        return top
