'''
Perform detection + tracking + recognition
Run: python3 generic_detection_tracking.py -c <camera_path> default <rabbit_mq>
                                                                    (for reading frames)
                                           -a <area> default 'None'
                                                                    (for area)
                                           -wi True default False
                                                                    (for writing all face-tracks)
                                           -vo True default False
                                                                    (for write tracking video)
'''
import argparse
import configparser
import time
import cv2
import numpy as np
import os
import logging
import glob
import json
import threading
from queue import Queue
from pymongo import MongoClient
from merge_split_utils import split_merge_id
from frame_reader import URLFrameReader, RabbitFrameReader
# from face_detector import MTCNNDetector
# from face_extractor import FacenetExtractor
from matcher import FaissMatcher, SVMMatcher
# from tf_graph import FaceGraph
from config import Config
from rabbitmq import RabbitMQ
from video_writer import VideoHandle
from tracker_manager_2 import TrackerManager
from face_info import FaceInfo
# from preprocess import Preprocessor, align_and_crop
# from face_align import AlignCustom
from cv_utils import (FaceAngleUtils, CropperUtils, is_inner_bb,
                      clear_session_folder, create_if_not_exist, PickleUtils)
import gc
import sys
from frame_queue import FrameQueue
from database import DashboardDatabase
from tensorflow_adapter import TensorflowAdapter
rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

# rabbit_mq.channel.queue_declare(queue=Config.Queues.MERGE)
# rabbit_mq.channel.queue_declare(queue=Config.Queues.SPLIT)
# rabbit_mq.channel.queue_declare(queue=Config.Queues.ACTION)
rabbit_mq.channel.queue_declare(queue='pdl-result')
TensorflowAdapter()
# mongodb_client = MongoClient(
#     Config.MongoDB.IP_ADDRESS,
#     Config.MongoDB.PORT,
#     username=Config.MongoDB.USERNAME,
#     password=Config.MongoDB.PASSWORD,
#     socketKeepAlive=True)
# mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
# mongodb_dashinfo = mongodb_db['pdl-dashinfo']
# mongodb_faceinfo = mongodb_db['pdl-faceinfo']
# mongodb_mslog = mongodb_db['pdl-mslog']

database = DashboardDatabase()


def generic_function(cam_url, queue_reader, area, re_source, use_frame_queue):
    global rabbit_mq
    '''
    This is main function
    '''
    print("Generic function")
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))

    if Config.Matcher.CLEAR_SESSION:
        clear_session_folder()

    if Config.Mode.CALC_FPS:
        start_time = time.time()

    if cam_url is not None:
        frame_reader = URLFrameReader(
            cam_url, scale_factor=1, re_source=re_source)
    elif queue_reader is not None:
        frame_reader = RabbitFrameReader(rabbit_mq, queue_reader)
    else:
        print('Empty Image Source')
        return -1

    if use_frame_queue:
        frame_src = FrameQueue(
            frame_reader, max_queue_size=Config.Frame.FRAME_QUEUE_SIZE)
        frame_src.start()
    else:
        frame_src = frame_reader

    video_out = None
    if Config.Track.TRACKING_VIDEO_OUT:
        video_out = VideoHandle(time.time(), video_out_fps, int(video_out_w),
                                int(video_out_h))

    # Variables for tracking faces
    frame_counter = 0
    # Variables holding the correlation trackers and the name per faceid
    tracking_dirs = os.listdir(Config.Dir.TRACKING_DIR)
    if tracking_dirs == []:
        current_tracker_id = 0
    else:
        list_of_trackid = [int(tracking_dir) for tracking_dir in tracking_dirs]
        current_tracker_id = max(list_of_trackid) + 1
    imageid_to_keyid = {}

    matcher = FaissMatcher()
    matcher.build(
        database, imageid_to_keyid=imageid_to_keyid, use_image_id=True)
    tracker_manager = TrackerManager(
        area,
        matcher,
        database.mongodb_faceinfo,
        imageid_to_keyid=imageid_to_keyid,
        current_id=current_tracker_id)

    try:
        while True:
            frame = frame_src.next_frame()
            if Config.Mode.CALC_FPS:
                fps_counter = time.time()
            if frame is None:
                print("Waiting for the new image")
                tracker_manager.update(rabbit_mq)
                time.sleep(1)
                continue

            # track by kcf
            tracker_manager.update_trackers(frame)
            tracker_manager.update(rabbit_mq)

            origin_bbs, points = TensorflowAdapter.detect_face(frame)
            if len(origin_bbs) > 0:
                origin_bbs = [origin_bb[:4] for origin_bb in origin_bbs]
                embeddings_array = [None] * len(origin_bbs)
                tracker_manager.process_new_detections(
                    frame,
                    origin_bbs,
                    points,
                    embeddings_array,
                    frame_id=frame_counter)

            frame_counter += 1
            if Config.Mode.CALC_FPS:
                print("FPS: %f" % (1 / (time.time() - fps_counter)))

        #TODO: this line never run
        tracker_manager.update(rabbit_mq)

    except KeyboardInterrupt:
        if use_frame_queue:
            frame_src.stop()
        print('Keyboard Interrupt !!! Release All !!!')
        tracker_manager.update(rabbit_mq)

        frame_src.release()
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format(
                (frame_counter + 1) / (time.time() - start_time)))
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
            video_out.release()

    else:
        raise Exception('Error')


def strbool(v):
    if strbool == True:
        return True
    if strbool == False:
        return False
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments(argv, config, env):
    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c',
        '--cam_url',
        help='your camera ip address',
        default=config[env]['cam_url'])
    parser.add_argument(
        '-a',
        '--area',
        help='The area that your ip camera is recording at',
        default=config[env]['area'])
    parser.add_argument(
        '-vo',
        '--video_out',
        help='Write tracking video out following the path data/tracking',
        default=None)
    parser.add_argument(
        '-db',
        '--dashboard',
        help='Send dashboard result',
        action='store_true',
        default=strbool(config[env]['dashboard']))
    parser.add_argument(
        '-cs',
        '--clear_session',
        help='write tracking folder with good element n min distance',
        action='store_true')
    parser.add_argument(
        '-cq', '--queue_reader', help='read frame from queue', default=None)
    #     parser.add_argument('-fem',
    #                         '--face_extractor_model',
    #                         help='path to model want to use instead of default',
    #                         default=Config.Model.FACENET_DIR)
    parser.add_argument(
        '-rs',
        '--re_source',
        help='Set the stream source again if stream connection is interrupted',
        action='store_true',
        default=strbool(config[env]['re_source']))
    parser.add_argument(
        '-rb',
        '--rethinkdb',
        help=
        'Use the first 30 face images to recognize and send it to rethinkdb',
        action='store_true')
    parser.add_argument(
        '-ufq',
        '--use_frame_queue',
        help='use_frame_queue mode',
        action='store_true',
        default=strbool(config[env]['use_frame_queue']))
    return parser.parse_args()


def main(args):
    # Run

    # if args.video_out is not None:
    #     Config.Track.TRACKING_VIDEO_OUT = True
    # Config.Track.VIDEO_OUT_PATH = args.video_out
    Config.SEND_QUEUE_TO_DASHBOARD = args.dashboard
    Config.Matcher.CLEAR_SESSION = args.clear_session
    Config.Track.SEND_RECOG_API = args.rethinkdb

    generic_function(args.cam_url, args.queue_reader, args.area, args.re_source,
                     args.use_frame_queue)


if __name__ == '__main__':
    # Main steps
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    env = ""
    try:
        env = os.environ["EYEQ_PROD"]
    except:
        pass
    config = configparser.ConfigParser()
    config.read("./prod/deploy_args.ini")
    print("env" + env)
    if env not in config.sections():
        env = "default"
        print(
            "[ERROR] Please assign the correct production name to the EYEQ_PROD environment variable, or the default configuration will be used."
        )
        print("Available productions: {}".format(config.sections()))
    else:
        print("[INFO] Running {} production.".format(env))

    main(parse_arguments(sys.argv[1:], config, env))
