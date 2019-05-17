import argparse
import time
import numpy as np
import os
import logging
from frame_reader import URLFrameReader, RabbitFrameReader
from config import Config
from video_writer import VideoHandle
from cv_utils import (FaceAngleUtils, CropperUtils, is_inner_bb,
                      clear_session_folder, create_if_not_exist)
from datetime import datetime
from rabbitmq import RabbitMQ
import json
from database import DashboardDatabase
import pipe
'''
        SplitMergeThread, \
        FindSimilarFaceThread
        '''
from matcher import KdTreeMatcher
from timer import Timer
import logging


def generic_function(cam_url, queue_reader, area, face_extractor_model,
                     re_source):
    '''
    This is main function
    '''
    print("Generic function")
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))

    # TODO: init logger, modulize this?

    # Variables for tracking faces
    frame_counter = 0

    if Config.Matcher.CLEAR_SESSION:
        clear_session_folder()

    if Config.Mode.CALC_FPS:
        start_time = time.time()
    if cam_url is not None:
        frame_reader = URLFrameReader(cam_url)
    else:
        print('Empty Image Source')
        return -1

    video_out_fps, video_out_w, video_out_h, = frame_reader.get_info()
    print(video_out_fps, video_out_w, video_out_h)

    video_out = None
    if Config.Track.TRACKING_VIDEO_OUT:
        video_out = VideoHandle(time.time(), video_out_fps, int(video_out_w),
                                int(video_out_h))

    db = DashboardDatabase(use_image_id=True)
    rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                         (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))
    matcher = KdTreeMatcher()
    matcher.build(db)

    # find current track
    import glob
    tracking_dirs = glob.glob(Config.Dir.TRACKING_DIR + '/*')

    if tracking_dirs == []:
        number_of_existing_trackers = 0
    else:
        lof_int_trackid = [
            int(tracking_dir.split('/')[-1]) for tracking_dir in tracking_dirs
        ]
        number_of_existing_trackers = max(lof_int_trackid) + 1

    mode = 'video'  # video, live
    '''
    # Feature 1: Find Merge Split
    splitMerge = pipe.SplitMergeThread(database=db, rabbit_mq=rabbit_mq, matcher=matcher)
    splitMerge.daemon = True
    splitMerge.start()

    # Feature 2: Find similar
    findSimilarFaceThread = pipe.FindSimilarFaceThread(database=db, rabbit_mq=rabbit_mq)
    findSimilarFaceThread.daemon = True
    findSimilarFaceThread.start()
    '''

    # main program stage
    stageDetectFace = pipe.Stage(pipe.FaceDetectWorker, 1)
    stagePreprocess = pipe.Stage(pipe.PreprocessDetectedFaceWorker, 1)
    stageDistributor = pipe.Stage(pipe.FaceDistributorWorker, 1)
    stageExtract = pipe.Stage(pipe.FaceExtractWorker, 1)
    stageTrack = pipe.Stage(
        pipe.FullTrackTrackingWorker,
        1,
        area=area,
        database=db,
        matcher=matcher,
        init_tracker_id=number_of_existing_trackers)
    stageResultToTCH = pipe.Stage(
        pipe.SendToDashboardWorker, 1, database=db, rabbit_mq=rabbit_mq)
    stageStorage = pipe.Stage(pipe.DashboardStorageWorker, 1)
    stageDatabase = pipe.Stage(pipe.DashboardDatabaseWorker, 1, database=db)

    stageDetectFace.link(stagePreprocess)
    stagePreprocess.link(stageDistributor)
    stageDistributor.link(stageExtract)
    stageExtract.link(stageTrack)
    stageTrack.link(stageResultToTCH)
    stageTrack.link(stageStorage)
    stageTrack.link(stageDatabase)

    if Config.Track.TRACKING_VIDEO_OUT:
        stageVideoOut = pipe.Stage(
            pipe.VideoWriterWorker, 1, database=db, video_out=video_out)
        stageTrack.link(stageVideoOut)

    pipeline = pipe.Pipeline(stageDetectFace)

    print('Begin')
    try:
        while frame_reader.has_next():
            #continue
            frame = frame_reader.next_frame()
            if frame is None:
                if mode == 'video':
                    print("Wait for executor to finish it jobs")
                    pipeline.put(None)
                    break
                if mode == 'live':
                    if re_source:
                        print('Trying to connect the stream again ...')
                        if cam_url is not None:
                            frame_reader = URLFrameReader(
                                cam_url, scale_factor=1, should_crop=True)
                    continue

            print('Read frame', frame_counter, frame.shape)

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                # timer = Timer(frame_counter)
                task = pipe.Task(pipe.Task.Frame)
                task.package(frame=frame, frame_info=frame_counter)
                pipeline.put(task)
                # pipeline.put((frame, frame_counter, timer))

            frame_counter += 1

        print('Time elapsed: {}'.format(time.time() - start_time))
        print('Avg FPS: {}'.format(
            (frame_counter + 1) / (time.time() - start_time)))
        frame_reader.release()
        '''
        splitMerge.join()
        findSimilarFaceThread.join()
        '''

    except KeyboardInterrupt:
        if Config.Track.TRACKING_VIDEO_OUT:
            video_out.release_tmp()
        pipeline.put(None)
        print('Keyboard Interrupt !!! Release All !!!')
        print('Time elapsed: {}'.format(time.time() - start_time))
        print('Avg FPS: {}'.format(
            (frame_counter + 1) / (time.time() - start_time)))
        frame_reader.release()
        '''
        splitMerge.join()
        findSimilarFaceThread.join()
        '''


if __name__ == '__main__':
    # Main steps
    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--cam_url', help='your camera ip address', default=None)
    parser.add_argument(
        '-a',
        '--area',
        help='The area that your ip camera is recording at',
        default='TCH')
    parser.add_argument(
        '-vo',
        '--video_out',
        help='Write tracking video out following the path data/tracking',
        default=None)
    parser.add_argument(
        '-db', '--dashboard', help='Send dashboard result', action='store_true')
    parser.add_argument(
        '-cs',
        '--clear_session',
        help='write tracking folder with good element n min distance',
        action='store_true')
    parser.add_argument(
        '-cq', '--queue_reader', help='read frame from queue', default=None)
    parser.add_argument(
        '-fem',
        '--face_extractor_model',
        help='path to model want to use instead of default',
        default=Config.Model.FACENET_DIR)
    parser.add_argument(
        '-rs',
        '--re_source',
        help='Set the stream source again if stream connection is interrupted',
        action='store_true')
    parser.add_argument(
        '-rb',
        '--rethinkdb',
        help=
        'Use the first 30 face images to recognize and send it to rethinkdb',
        action='store_true')
    args = parser.parse_args()

    # Run
    if args.video_out is not None:
        Config.Track.TRACKING_VIDEO_OUT = True
        Config.Track.VIDEO_OUT_PATH = args.video_out
    Config.Mode.SEND_QUEUE_TO_DASHBOARD = args.dashboard
    Config.Matcher.CLEAR_SESSION = args.clear_session
    Config.Track.SEND_RECOG_API = args.rethinkdb

    create_if_not_exist(Config.Dir.TRACKING_DIR)
    create_if_not_exist(Config.Dir.LOG_DIR)

    generic_function(args.cam_url, args.queue_reader, args.area,
                     args.face_extractor_model, args.re_source)
