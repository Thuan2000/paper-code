import os
import cv2
from scipy import ndimage
from ems import base_server
from pipe import pipeline, task, stage
from worker import face_detect_worker, face_extract_worker
from worker import face_verification_worker
from core import face_detector, tf_graph
from config import Config


class FaceVerificationServer(base_server.AbstractServer):

    def init(self):
        self.pipeline = self.build_pipeline()
        self.mtcnn_detector = face_detector.MTCNNDetector(tf_graph.FaceGraph())
        pass

    def add_endpoint(self):
        self.app.add_url_rule('/verify', 'verify', self.face_verification, methods=['POST'])

    def face_verification(self):
        images_1st, images_2nd = self.request_data_parser()
        if images_1st and images_2nd:
            images_1st = self.calibrate_angle(images_1st)
            images_2nd = self.calibrate_angle(images_2nd)
            for image in images_1st:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=image, frame_info='images_1st')
                self.pipeline.put(_task)

            for image in images_2nd:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=image, frame_info='images_2nd')
                self.pipeline.put(_task)

            _task = task.Task(task.Task.Event)
            _task.package(frame_info_1='images_1st', frame_info_2='images_2nd')
            self.pipeline.put(_task)
            # get result from pipeline
            data = self.pipeline.get()
            status = data.pop('status')
            if status == Config.Status.SUCCESSFUL:
                return self.response_success(data)
            else:
                message = 'Faces not found'
                return self.response_error(message)
        else:
            return self.response_error('both input field must not be empty')

    def request_data_parser(self):
        images = []
        files_dict = self.request.files.to_dict(flat=False)
        for value in files_dict.values():
            _images = []
            file_ext = os.path.splitext(value[0].filename)[1].lower()
            if file_ext in Config.EXTENSION.VIDEO:
                for video in value:
                    save_path = os.path.join(os.getcwd(), video.filename)
                    video.save(save_path)
                    cap = cv2.VideoCapture(save_path)
                    ret, frame = cap.read()
                    while ret:
                        _images.append(frame)
                        ret, frame = cap.read()
                    os.remove(save_path)
                images.append(_images)
            elif file_ext in Config.EXTENSION.IMAGE:
                for image in value:
                    save_path = os.path.join(os.getcwd(), image.filename)
                    image.save(save_path)
                    _image = cv2.imread(save_path)
                    _images.append(_image)
                    os.remove(save_path)
                images.append(_images)
            else:
                images.append([])

        if len(images) >= 2:
            images_cvted_1 = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images[0]]
            images_cvted_2 = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images[1]]
            return images_cvted_1, images_cvted_2
        return [], []

    def build_pipeline(self):
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_verification_worker.PreprocessDetectedFaceWorker, 1)
        stageCollect = stage.Stage(face_verification_worker.CollectWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.EmbeddingExtractWorker, 1)
        stageMatching = stage.Stage(face_verification_worker.MatchingWorker, 1)

        stageDetectFace.link(stagePreprocess)
        stagePreprocess.link(stageCollect)
        stageCollect.link(stageExtract)
        stageExtract.link(stageMatching)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline

    def calibrate_angle(self, images):
        '''
        run on mtcnn and rotate images until find the right angle: 0, 90, 180, 270
        '''
        angles = [0, 90, 180, 270]
        slice_step = max(int(len(images)/3), 1)
        test_images = images[::slice_step]
        for angle in angles:
            detected_face_count = 0
            rotated_images = [ndimage.rotate(_image, angle) for _image in test_images]
            for image in rotated_images:
                bbox, _ = self.mtcnn_detector.detect_face(image)
                if bbox.size > 0:
                    detected_face_count += 1
            if detected_face_count/len(test_images) > 0.5:
                return rotated_images
        return images
