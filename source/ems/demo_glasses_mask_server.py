import argparse
import numpy as np
import threading
import time
import os
from pipe import pipeline, stage, task
from utils import dict_and_list
from utils import socket_client
from core import frame_reader, mask_glasses, preprocess
from core import face_detector, tf_graph
from ems import base_server
from utils.logger import logger
from config import Config


class DemoGlassesMaskServer(base_server.AbstractServer):
    MAX_RETRY = 10

    def init(self):

        # TODO: Remove hardcode
        hostname = 'webapp-annotation-cv_webapp_1'
        port = 3000
        self.image_socket = socket_client.ImageSocket(hostname, port)
        self.frame_reader = frame_reader.SocketIOFrameReader(self.image_socket)

        threading.Thread(target=self.wait_for_new_demo).start()

    def wait_for_new_demo(self):
        # first call will start the demo any way. When the demo stop,
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

        # TODO: Move this to pipeline
        _face_detector = face_detector.MTCNNDetector(tf_graph.FaceGraph())
        _preprocessor = preprocess.Preprocessor(algs=preprocess.normalization)
        glasses_classifier = mask_glasses.GlassesClassifier()
        mask_classifier = mask_glasses.MaskClassifier()
        waiting_images = dict_and_list.WaitingImageQueue(2)
        result_list = dict_and_list.EmbsDict(4)

        print('Begin')
        retry = 0
        frame_counter = 0
        while True:
            frame, client_id = self.frame_reader.next_frame()
            if frame is None:
                retry += 1
                print('Retry %s times' % retry)

                # TODO: a bang want this to run forever, maybe remove in the furture
                #if retry >= DemoGlassesMaskServer.MAX_RETRY:
                #    break
                continue

            # TODO: Try to convert to pipeline
            frame_counter += 1
            if frame_counter % 2 == 0:
                # skip 1 frame every
                continue

            print('Frame', frame_counter, frame.shape)
            bbs, pts = _face_detector.detect_face(frame)
            if len(bbs) > 0:
                preprocessed_frame = _preprocessor.process(frame)
                waiting_images.put(client_id, preprocessed_frame)
                if waiting_images.has_enough(client_id):

                    images = waiting_images.get(client_id)
                    images = np.array(images)

                    has_masks = mask_classifier.is_wearing_mask(images)
                    has_glasses = glasses_classifier.is_wearing_glasses(images)

                    results = np.array([has_glasses, has_masks])
                    result_list.put(client_id, results.T)

                    matching_results = result_list.get(client_id)
                    has_glasses = self.find_most_common(
                        matching_results[:, 0])
                    has_mask = self.find_most_common(
                        matching_results[:, 1])
                    result = 'Has glasses %s, has mask %s' % (has_glasses,
                                                              has_mask)
                    print('Result', result)
                    self.image_socket.put_result(
                        client_id=client_id,
                        status='successful',
                        face_name=result)
            else:
                self.image_socket.put_result(
                    client_id=client_id, status='alert', face_name='Detecting')

        print("No more frame, stop")

    def find_most_common(self, ls):
        # TODO: Move to utils
        try:
            top = mode(ls)
        except:
            top = ls[0]
        return top

        # task = pipe.Task(pipe.Task.Frame)
        # task.package(frame=frame, frame_info=client_id)
        # pipeline.put(task)

    # def build_pipeline(image_socket):
    #     stageDetectFace = pipe.Stage(pipe.FaceDetectWorker, 1)
    #     stagePreprocess = pipe.Stage(pipe.PreprocessFrameWorker, 1)
    #     stageExtract = pipe.Stage(pipe.GlassesMaskExtractWorker, 1)
    #     stageMatching = pipe.Stage(pipe.GlassesMaskMatchingWorker, socket=image_socket)

    #     stageDetectFace.link(stagePreprocess)
    #     stagePreprocess.link(stageExtract)
    #     stageExtract.link(stageMatching)

    #     pipeline = pipe.Pipeline(stageDetectFace)
    #     return pipeline
