import numpy as np
import time
from pipe import worker, task
from pipe.trace_back import process_traceback
from core import preprocess
from core.cv_utils import CropperUtils
from core.tracking import detection
from utils.logger import logger
from config import Config
from core.face_align import AlignCustom


class PreprocessDetectedFaceWorker(worker.Worker):
    '''
    Apply preprocess to all faces
    Input: bbs (bouncing boxes), pts (landmarks), frame, frame_info
    Output:
    - faces: crop from frame,
    - preprocessed_images: np array, ready to extract features
    - frame, frame_info
    '''

    def doInit(self):
        self.preprocessor = preprocess.Preprocessor()
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # start = time.time()
        # TODO: Get timer
        data = _task.depackage()
        bbs, pts, frame, frame_info = data['bbs'], data['pts'], data[
            'frame'], data['frame_info']

        nrof_faces = len(bbs)
        # timer.preprocess_start()
        faces = []
        preprocessed_images = []
        for i in range(nrof_faces):
            display_face, padded_bb_str = CropperUtils.crop_display_face(
                frame, bbs[i][:-1])
            face = detection.FaceInfo(bbs[i][:-1], bbs[i][-1], frame_info, display_face,
                            padded_bb_str, pts[:, i])
            cropped_face = CropperUtils.crop_face(frame, bbs[i][:-1])
            preprocessed = self.preprocessor.process(cropped_face)
            preprocessed_images.append(preprocessed)
            faces.append(face)
        preprocessed_images = np.array(preprocessed_images)
        # timer.preprocess_done()
        _task = task.Task(task.Task.Face)
        _task.package(
            faces=faces,
            images=preprocessed_images,
            frame=frame,
            frame_info=frame_info)
        self.putResult(_task)
        # print(self.name, time.time() - start)


class PreprocessDetectedArcFaceWorker(worker.Worker):
    '''
    Apply preprocess to all faces
    Input: bbs (bouncing boxes), pts (landmarks), frame, frame_info
    Output:
    - faces: crop from frame,
    - preprocessed_images: np array, ready to extract features
    - frame, frame_info
    '''

    def doInit(self, use_coeff_filter=True):
        self.face_preprocessor = preprocess.Preprocessor(preprocess.align_and_crop)
        self.use_coeff_filter = use_coeff_filter
        if use_coeff_filter:
            self.coeff_preprocessor = preprocess.Preprocessor()

        self.aligner = AlignCustom()
        self.prewhitening = False
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # start = time.time()
        # TODO: Get timer
        data = _task.depackage()
        bbs, pts, frame, frame_info = data['bbs'], data['pts'], data[
            'frame'], data['frame_info']

        nrof_faces = len(bbs)
        # timer.preprocess_start()
        faces = []
        preprocessed_face_images = []

        preprocessed_coeff_images = []
        for i in range(nrof_faces):
            display_face, padded_bb_str = CropperUtils.crop_display_face(
                frame, bbs[i][:-1])
            face = detection.FaceInfo(bbs[i][:-1], bbs[i][-1], frame_info, display_face,
                            padded_bb_str, pts[:, i])
            preprocessed_face = self.face_preprocessor.process(frame, pts[:, i], self.aligner, Config.Align.IMAGE_SIZE, self.prewhitening)
            preprocessed_face_images.append(preprocessed_face)
            faces.append(face)

            if self.use_coeff_filter:
                cropped_face = CropperUtils.crop_face(frame, bbs[i][:-1])
                preprocessed_coeff = self.coeff_preprocessor.process(cropped_face)
                preprocessed_coeff_images.append(preprocessed_coeff)

        preprocessed_face_images = np.array(preprocessed_face_images)

        preprocessed_coeff_images = np.array(preprocessed_coeff_images)

        # timer.preprocess_done()
        _task = task.Task(task.Task.Face)
        _task.package(
            faces=faces,
            images=preprocessed_face_images,
            coeff_images=preprocessed_coeff_images,
            frame=frame,
            frame_info=frame_info)
        self.putResult(_task)
        # print(self.name, time.time() - start)
