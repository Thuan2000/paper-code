import argparse
import os
import time
import numpy as np
import imageio
from core import matcher
from utils import database
from ems import base_server
from utils.logger import logger
from core.cv_utils import base64str_to_frame
from pipe import pipeline, stage, task
from worker import face_detect_worker, face_preprocess_worker
from worker import face_extract_worker, demo_ecommerce_worker
from config import Config


class DemoEcommerceServer(base_server.AbstractServer):

    def init(self):
        imageio.plugins.freeimage.download()
        self.database = database.EcommerceDatabase()
        self.matcher = matcher.KdTreeMatcher()
        self.matcher.build(self.database)
        self.pipeline = self.build_pipeline()

    def add_endpoint(self):
        self.app.add_url_rule(
            '/register', 'register', self.register_api, methods=['POST'])

    def register_api(self):
        frame_info = 'Demo'

        images_str = self.request.json['images']
        for image_str in images_str:
            success, frame = base64str_to_frame(image_str)
            if success:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_info)
                self.pipeline.put(_task)

        # notify pipeline there is no more images and
        _task = task.Task(task.Task.Event)
        _task.package(client_id=frame_info)
        self.pipeline.put(_task)

        results = self.pipeline.results()
        face_id = next(results)
        response = DemoEcommerceServer.response_success({'faceId': face_id})
        return response

    def build_pipeline(self):
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_preprocess_worker.PreprocessDetectedFaceWorker, 1)
        stageCollect = stage.Stage(demo_ecommerce_worker.EcommerceCollectWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.EmbeddingExtractWorker, 1)
        stageMatching = stage.Stage(
            demo_ecommerce_worker.EcommerceMatchingWorker,
            size=1,
            database=self.database,
            matcher=self.matcher)

        stageDetectFace.link(stagePreprocess)
        stagePreprocess.link(stageCollect)
        stageCollect.link(stageExtract)
        stageExtract.link(stageMatching)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline
