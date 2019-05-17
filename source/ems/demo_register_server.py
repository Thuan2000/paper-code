import argparse
import os
import threading
import queue
import time
from ems import base_server
from core import frame_reader, matcher
from utils import database, socket_client
from pipe import pipeline, stage, task
from worker import face_detect_worker, face_preprocess_worker
from worker import face_extract_worker, demo_register_worker
from utils.logger import logger
from config import Config


class DemoRegisterServer(base_server.AbstractServer):
    MAX_RETRY = 10

    def init(self):
        print('Demo register server')
        self.register_queue = queue.Queue()
        self.database = database.DemoDatabase()

        # TODO: Remove hard code here
        hostname = 'webapp-annotation-cv_webapp_1'
        port = 3000
        self.image_socket = socket_client.ImageSocket(hostname, port)
        self.frame_reader = frame_reader.SocketIOFrameReader(self.image_socket)

        self.matcher = matcher.KdTreeMatcher()
        self.matcher.build(self.database)

        threading.Thread(target=self.wait_for_new_demo).start()

    def add_endpoint(self):
        self.app.add_url_rule(
            '/register', 'register', self.register_api, methods=['POST'])

    def wait_for_new_demo(self):
        # first call will start the demo any way. Ưhen the demo stop,
        # terminate the process to reclaim resouce and start to wait for new demo to come
        self.start_demo()

        while True:
            frame, _ = self.frame_reader.next_frame()
            if frame is not None:
                self.start_demo()
            else:
                print('Wating for new demo')
                time.sleep(10)

    def start_demo(self):
        demo_thread = threading.Thread(target=self.run_demo)
        demo_thread.start()
        demo_thread.join()

    def run_demo(self):
        '''
        This is main function
        '''
        _pipeline = self.build_pipeline()

        print('Begin')
        retry = 0
        while True:
            try:
                face, client_id, send_at = self.register_queue.get(False)
                _task = task.Task(task.Task.Event)
                _task.package(face=face, client_id=client_id, sent_at=send_at)
            except:
                frame, frame_info = self.frame_reader.next_frame()

                if frame is None:
                    retry += 1
                    print('Retry %s times' % retry)
                    if retry > DemoRegisterServer.MAX_RETRY:
                        break
                    continue
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_info)
            _pipeline.put(_task)

        print("No more frame, stop")
        _pipeline.put(None)

    def register_api(self):
        logger.debug(self.request.json)
        face = self.request.json.get('face')
        client_id = self.request.json.get('client_socket_id')
        send_at = self.request.json.get('sent_at')
        logger.debug('POST /register name: %s, client: %s, timestamp: %s' %
                     (face, client_id, send_at))
        self.register_queue.put((face, client_id, send_at))
        return DemoRegisterServer.response_success('Registering')

    def build_pipeline(self):
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_preprocess_worker.PreprocessDetectedFaceWorker, 1)
        stageCollect = stage.Stage(demo_register_worker.RegisterCollectWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.EmbeddingExtractWorker, 1)
        stageMatching = stage.Stage(
            demo_register_worker.RegisterMatchingWorker,
            size=1,
            database=self.database,
            matcher=self.matcher,
            socket=self.image_socket)

        stageDetectFace.link(stagePreprocess)
        stagePreprocess.link(stageCollect)
        stageCollect.link(stageExtract)
        stageExtract.link(stageMatching)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline
