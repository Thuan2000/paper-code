import time
import numpy as np
from core import face_extractor, tf_graph, preprocess
from pipe import worker, task
from pipe.trace_back import process_traceback
from core.cv_utils import CropperUtils
from utils.logger import logger
from config import Config


class FaceExtractWorker(worker.Worker):
    '''
    Extract features from faces, do multi-faces at a time to reduce time
    Input: faces, preprocessed_images, frame, frame_info
    Output: A single face at a time for tracking
    - face, frame, frame_info
    '''

    # 936.0 MiB
    def doInit(self):
        face_graph = tf_graph.FaceGraph()
        try:
            self.extractor = face_extractor.FacenetExtractor(
                face_graph, model_path=Config.Model.COEFF_DIR)
        except:
            logger.exception("CUDA out off memory", exc_info=True)

        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        faces, preprocessed_images, frame, frame_info = data['faces'], data[
            'images'], data['frame'], data['frame_info']
        logger.info("Extract: %s, #Faces: %s" % (frame_info, len(faces)))

        # timer.extract_start()
        embs, coeffs = self.extractor.extract_features(preprocessed_images)
        # timer.extract_done()

        is_put = False
        for i, face in enumerate(faces):
            face.embedding = embs[i, :]
            face.set_face_quality(coeffs[i])
            if face.is_good():
                _task = task.Task(task.Task.Face)
                _task.package(face=face, frame=frame, frame_info=frame_info)
                self.putResult(_task)
                is_put = True

        if not is_put:
            _task = task.Task(task.Task.Frame)
            _task.package(frame=frame, frame_info=frame_info)
            self.putResult(_task)


class EmbeddingExtractWorker(worker.Worker):
    '''
    Extract features from faces, do multi-faces at a time to reduce time
    Input: faces, preprocessed_images, frame, frame_info
    Output: embs extracted from frame
    - face, frame, frame_info
    '''

    # 936.0 MiB

    def doInit(self):
        face_graph = tf_graph.FaceGraph()
        try:
            self.extractor = face_extractor.FacenetExtractor(
                face_graph, model_path=Config.Model.COEFF_DIR)
        except:
            logger.exception("CUDA out off memory", exc_info=True)

        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        images, frame_info = data['images'], data['frame_info']

        embs = np.array([])
        if images.size > 0:
            try:
                embs, _ = self.extractor.extract_features(images)
            except ValueError:
                print('Can not extract image with shape', images.shape)
        _task = task.Task(task.Task.Face)
        _task.package(embs=embs, frame_info=frame_info)
        self.putResult(_task)


class BatchEmbeddingExtractWorker(worker.Worker):

    def doInit(self):
        face_graph = tf_graph.FaceGraph()
        try:
            self.extractor = face_extractor.FacenetExtractor(
                face_graph, model_path=Config.Model.COEFF_DIR)
        except:
            logger.exception("CUDA out off memory", exc_info=True)
        self.preprocessor = preprocess.Preprocessor()
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        start = time.time()
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_EXTRACTION:
            return

        tracker = data['tracker']
        nrof_elements = len(tracker.elements)
        # here we do the batch embs extracting
        _interval = Config.Track.NUM_IMAGE_PER_EXTRACT
        for i in range(0, nrof_elements, _interval):
            _interval = min(nrof_elements - i, _interval)
            preprocessed_images = []
            for j in range(i, i + _interval):
                face_image = CropperUtils.reverse_display_face(
                    tracker.elements[j].face_image, tracker.elements[j].str_padded_bbox)
                preprocessed_image = self.preprocessor.process(face_image)
                preprocessed_images.append(preprocessed_image)
            embeddings_array, _ = self.extractor.extract_features_all_at_once(
                preprocessed_images)
            tracker.update_embeddings(embeddings_array, i, _interval)
        _task = task.Task(task.Task.Face)
        _task.package(tracker=tracker)
        self.putResult(_task)
        print(self.name, time.time() - start)



class MultiFacesExtractWorker(worker.Worker):
    '''
    Extract features from faces, do multi-faces at a time to reduce time
    Input: faces, preprocessed_images, frame, frame_info
    Output: A single face at a time for tracking
    - face, frame, frame_info
    '''

    # 936.0 MiB
    def doInit(self, use_coeff_filter=True):
        try:
            facenet_graph = tf_graph.FaceGraph()
            self.embs_extractor = face_extractor.FacenetExtractor(
                facenet_graph, model_path=Config.Model.FACENET_DIR)
            self.use_coeff_filter = use_coeff_filter
            if use_coeff_filter:
                coeff_graph = tf_graph.FaceGraph()
                self.coeff_extractor = face_extractor.FacenetExtractor(
                    coeff_graph, model_path=Config.Model.COEFF_DIR)
        except:
            logger.exception("CUDA out off memory", exc_info=True)
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        faces, preprocessed_images, frame, frame_info = data['faces'], data[
            'images'], data['frame'], data['frame_info']
        logger.info("Extract: %s, #Faces: %s" % (frame_info, len(faces)))

        # timer.extract_start()
        # TODO: we can use only one extracter
        face_infos = []
        if preprocessed_images.any():
            embs, _ = self.embs_extractor.extract_features_all_at_once(preprocessed_images)
            coeffs = [100]*embs.shape[0]
            if self.use_coeff_filter:
                _, coeffs = self.coeff_extractor.extract_features_all_at_once(preprocessed_images)
            # timer.extract_done()
            for i, face in enumerate(faces):
                face.embedding = embs[i, :]
                face.set_face_quality(coeffs[i])
                face_infos.append(face)

        _task = task.Task(task.Task.Face)
        _task.package(faces=face_infos)
        self.putResult(_task)


class MultiArcFacesExtractWorker(worker.Worker):
    '''
    Extract features from faces, do multi-faces at a time to reduce time
    Input: faces, preprocessed_images, frame, frame_info
    Output: A single face at a time for tracking
    - face, frame, frame_info
    '''

    # 936.0 MiB
    def doInit(self, use_coeff_filter=True):
        try:
            self.embs_extractor = face_extractor.ArcFaceExtractor(model_path=Config.Model.ARCFACE_DIR)
            self.use_coeff_filter = use_coeff_filter
            if use_coeff_filter:
                coeff_graph = tf_graph.FaceGraph()
                self.coeff_extractor = face_extractor.FacenetExtractor(
                    coeff_graph, model_path=Config.Model.COEFF_DIR)
        except:
            logger.exception("CUDA out off memory", exc_info=True)
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        faces, preprocessed_images, preprocessed_coeff_images, frame, frame_info = data['faces'], data[
            'images'], data['coeff_images'], data['frame'], data['frame_info']
        logger.info("Extract: %s, #Faces: %s" % (frame_info, len(faces)))

        # timer.extract_start()
        # TODO: we can use only one extracter
        face_infos = []
        if preprocessed_images.any():
            embs, _ = self.embs_extractor.extract_features_all_at_once(preprocessed_images)
            coeffs = [100]*embs.shape[0]
            if self.use_coeff_filter:
                _, coeffs = self.coeff_extractor.extract_features_all_at_once(preprocessed_coeff_images)
            # timer.extract_done()
            for i, face in enumerate(faces):
                face.embedding = embs[i, :]
                face.set_face_quality(coeffs[i])
                face_infos.append(face)

        _task = task.Task(task.Task.Face)
        _task.package(faces=face_infos)
        self.putResult(_task)


class ArcFacesEmbeddingExtractWorker(worker.Worker):
    '''
    Extract features from faces, do multi-faces at a time to reduce time
    Input: faces, preprocessed_images, frame, frame_info
    Output: embs extracted from frame
    - face, frame, frame_info
    '''

    # 936.0 MiB

    def doInit(self):
        face_graph = tf_graph.FaceGraph()
        try:
            self.extractor = face_extractor.ArcFaceExtractor(model_path=Config.Model.ARCFACE_DIR)

        except:
            logger.exception("CUDA out off memory", exc_info=True)

        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        images, frame_info = data['images'], data['frame_info']

        embs = np.array([])
        if images.size > 0:
            try:
                embs, _ = self.extractor.extract_features(images)
            except ValueError:
                print('Can not extract image with shape', images.shape)
        _task = task.Task(task.Task.Face)
        _task.package(embs=embs, frame_info=frame_info)
        self.putResult(_task)