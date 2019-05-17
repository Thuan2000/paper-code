import time
import threading
import queue
import cv2
from queue import Queue
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from matcher import KdTreeMatcher, FaissMatcher
from tf_graph import FaceGraph
from config import Config
from rabbitmq import RabbitMQ
from onetime.multi_client_generic_detection_tracking import generic_function
from cv_utils import decode_image, clear_session_folder
from preprocess import Preprocessor
from frame_reader import QueueFrameReader

# read config
if True:
    configs = []
    with open('../config.txt', 'r') as f:
        configs = f.readlines()
    configs = [txt_config.strip('\n') for txt_config in configs]
    Config.DEMO_FOR = configs[0]
    Config.Rabbit.IP_ADDRESS = configs[1]

face_rec_graph = FaceGraph()
face_extractor = FacenetExtractor(face_rec_graph, model_path=Config.FACENET_DIR)
detector = MTCNNDetector(face_rec_graph)
preprocessor = Preprocessor()
matcher = FaissMatcher()
matcher._match_case = 'TCH'
matcher.build(Config.REG_IMAGE_FACE_DICT_FILE)
rb = RabbitMQ()

frame_readers = dict()
register_command = dict()  # {session_id: [[register_name, video_path]]}
removed_sessions = Queue()
sent_msg_queue = Queue()
start_time = time.time()

while True:
    # if time.time() - start_time >= 10.0:
    #     try:
    #         while True:
    #             rm_id = removed_sessions.get(False)
    #             frame_readers.pop(rm_id, None)
    #             sessions.pop(rm_id, None)
    #             register_command.pop(rm_id, None)
    #             with open('multi_process_logging.txt', 'a') as f:
    #                 f.write('Kill no longer active session {}\n'.format(rm_id))
    #     except queue.Empty:
    #         pass
    #     start_time = time.time()

    command = rb.receive_once(queue_name='{}-command'.format(Config.DEMO_FOR))
    img_msg = rb.receive_once(queue_name='{}-live'.format(Config.DEMO_FOR))
    msg_list = rb.receive_once(queue_name='{}-register'.format(Config.DEMO_FOR))
    if msg_list is not None:
        print(msg_list)
        if msg_list[0] in register_command:
            register_command[msg_list[0]].put(msg_list[1:])
        else:
            register_command[msg_list[0]] = Queue()
            register_command[msg_list[0]].put(msg_list[1:])
    if command is not None:
        command = command[0]
        print(command)
        if command == 'flushdb':
            clear_session_folder()
            matcher.build(Config.REG_IMAGE_FACE_DICT_FILE)

    if msg_list is not None:
        sess_id = msg_list[0]
        if sess_id not in frame_readers:
            with open('multi_process_logging.txt', 'a') as f:
                f.write('Create new session {}\n'.format(sess_id))
            frame_readers[sess_id] = QueueFrameReader()
            register_command[sess_id] = Queue()
            register_command[sess_id].put(msg_list[1:])
            print('Create new session for register')
            thread = threading.Thread(
                target=generic_function,
                args=(
                    frame_readers,
                    'TCH',
                    sess_id,
                    detector,
                    face_extractor,
                    matcher,
                    register_command,
                    sent_msg_queue,
                ))
            thread.daemon = True
            thread.start()

    if img_msg is not None:
        # print(img_msg)
        sess_id = None
        sess_id = img_msg[0]
        frame_string = img_msg[1]
        frame = decode_image(frame_string)

        #cv2.imshow('frame', frame)
        #cv2.waitKey(1)
        # print(type(frame))
        if sess_id not in frame_readers:
            with open('multi_process_logging.txt', 'a') as f:
                f.write('Create new session {}\n'.format(sess_id))
            frame_readers[sess_id] = QueueFrameReader()
            frame_readers[sess_id].add_item(frame)
            register_command[sess_id] = Queue()
            thread = threading.Thread(
                target=generic_function,
                args=(
                    frame_readers,
                    'TCH',
                    sess_id,
                    detector,
                    face_extractor,
                    matcher,
                    register_command,
                    sent_msg_queue,
                ))
            thread.daemon = True
            thread.start()
        else:
            #             print('Send frame to session {}'.format(sess_id))
            frame_readers[sess_id].add_item(frame)

    if not sent_msg_queue.empty():
        queue_name, msg_content = sent_msg_queue.get()
        rb.send(queue_name, msg_content)
