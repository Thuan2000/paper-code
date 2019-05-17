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
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from matcher import FaissMatcher, SVMMatcher
from tf_graph import FaceGraph
from config import Config
from rabbitmq import RabbitMQ
from video_writer import VideoHandle
from tracker_manager import TrackerManager
from tracker_results_dict import TrackerResultsDict
from tracking_utils import FaceInfo
from preprocess import Preprocessor, align_and_crop
from face_align import AlignCustom
from cv_utils import (FaceAngleUtils, CropperUtils, is_inner_bb,
                      clear_session_folder, create_if_not_exist, PickleUtils)

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

rabbit_mq.channel.queue_declare(queue=Config.Queues.MERGE)
rabbit_mq.channel.queue_declare(queue=Config.Queues.SPLIT)
rabbit_mq.channel.queue_declare(queue=Config.Queues.ACTION)
rabbit_mq.channel.queue_declare(queue=Config.Queues.LIVE_RESULT)

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)
mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
mongodb_dashinfo = mongodb_db[Config.MongoDB.DASHINFO_COLS_NAME]
mongodb_faceinfo = mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]
mongodb_mslog = mongodb_db[Config.MongoDB.MSLOG_COLS_NAME]

lock = threading.Lock()
querying_top10_image_ids_queue = []


def handle_filters(point, coeff, face_info, preprocessed_image):
    is_good_face = True
    # Calculate angle
    yaw_angle = FaceAngleUtils.calc_angle(point)
    pitch_angle = FaceAngleUtils.calc_face_pitch(point)
    bbox_str = '_'.join(
        np.array(face_info.bounding_box[:-1], dtype=np.unicode).tolist())

    if abs(yaw_angle) > Config.Filters.YAW:
        # img_path = '../data/outofangle/yaw_{}_{}_{}.jpg'.format(face_info.frame_id,
        #                                                         bbox_str,
        #                                                         abs(yaw_angle))
        # cv2.imwrite(img_path,
        #             cv2.cvtColor(face_info.display_image, cv2.COLOR_BGR2RGB))
        is_good_face = False
    if abs(pitch_angle) > Config.Filters.PITCH:
        # img_path = '../data/outofangle/pitch_{}_{}_{}.jpg'.format(face_info.frame_id,
        #                                                           bbox_str,
        #                                                           abs(pitch_angle))
        # cv2.imwrite(img_path,
        #             cv2.cvtColor(face_info.display_image, cv2.COLOR_BGR2RGB))
        is_good_face = False

    _, coeff_score = coeff_extractor.extract_features(preprocessed_image)
    if coeff_score < Config.Filters.COEFF:
        img_path = '../data/notenoughcoeff/{}_{}_{}.jpg'.format(
            face_info.frame_id, bbox_str, coeff_score)
        cv2.imwrite(img_path,
                    cv2.cvtColor(face_info.display_image, cv2.COLOR_BGR2RGB))
        is_good_face = False
    # else:
    #     with open('../data/coeff_log.txt', 'a') as f:
    #         f.write('{}_{}_{}, coeff: {}\n'.format(bbox_str,
    #                                                face_info.frame_id,
    #                                                face_info.str_padded_bbox,
    #                                                coeff))
    return is_good_face


def extract_info_from_json(rbmq_msg):
    return_dict = json.loads(rbmq_msg)
    return_anchor = return_dict['visitorId']
    return_list = return_dict['images']
    return_list = [
        return_dir.split('/')[-1].replace('.jpg', '')
        for return_dir in return_list
    ]
    return return_anchor, return_list


def give_this_id_10_closest_ids():
    # init matcher
    with open('../data/top10querylog.txt', 'a') as f:
        f.write('TOP10 QUERY IS BEING IN PROCESS !!!\n\n')
    global querying_top10_image_ids_queue
    global mongodb_faceinfo
    global mongodb_dashinfo
    embs = []
    labels = []
    cursors = mongodb_dashinfo.find({})
    unique_labels = [cursor['represent_image_id'] for cursor in cursors]
    cursors = mongodb_faceinfo.find({'image_id': {'$in': unique_labels}})
    for cursor in cursors:
        embs.append(np.array(cursor['embedding']))
        labels.append(cursor['image_id'])
    nof_registered_image_ids = len(labels)
    matcher = FaissMatcher()
    matcher.fit(embs, labels)

    with open('../data/top10querylog.txt', 'a') as f:
        f.write('MATCHER BUILT!!!\n\n')

    while True:
        if nof_registered_image_ids < mongodb_dashinfo.find({}).count():
            nof_registered_image_ids = mongodb_dashinfo.find({}).count()
            print('[Query TOP10] Update new registered image_id ...')
            cursors = mongodb_dashinfo.find({
                'represent_image_id': {
                    '$nin': labels
                }
            })
            unique_labels = [cursor['represent_image_id'] for cursor in cursors]
            cursors = mongodb_faceinfo.find({
                'image_id': {
                    '$in': unique_labels
                }
            })
            adding_embs = []
            adding_labels = []
            for cursor in cursors:
                adding_embs.append(np.array(cursor['embedding']))
                adding_labels.append(cursor['image_id'])
            old_embs = embs
            old_labels = labels
            embs = old_embs + adding_embs
            labels = old_labels + adding_labels

            matcher.update(adding_embs, adding_labels)

        if not len(querying_top10_image_ids_queue) == 0:
            lock.acquire()
            queue_data = querying_top10_image_ids_queue.pop()
            lock.release()
            results = {}
            session_id = queue_data['sessionId']
            image_id = queue_data['imageId']
            print('[Query TOP10] image_id: ' + image_id)
            with open('../data/top10querylog.txt', 'a') as f:
                f.write('image_id: ' + image_id + '\n')

            cursors = mongodb_faceinfo.find({'image_id': image_id})
            if cursors.count() == 0:
                print('[Query TOP10] THIS QUERY IMAGE ID HAS YET TO REGISTER')
                with open('../data/top10querylog.txt', 'a') as f:
                    f.write('THIS QUERY IMAGE ID HAS YET TO REGISTER\n')
                face_id = mongodb_dashinfo.find({
                    'represent_image_id': image_id
                })[0]['face_id']
                unique_labels = [
                    cursor['represent_image_id']
                    for cursor in mongodb_dashinfo.find({
                        'face_id': face_id
                    })
                ]
                for label in unique_labels:
                    results[label] = '0'
            else:
                query_emb = cursors[0]['embedding']
                dists, inds = matcher._classifier.search(
                    np.array(query_emb).astype('float32'), k=15)
                dists = np.squeeze(dists)
                inds = np.squeeze(inds)
                top_match_ids = [labels[idx] for idx in inds]
                for i, top_match_id in enumerate(top_match_ids):
                    if i < 11 and top_match_id != image_id:
                        results[top_match_id] = str(dists[i])
            msg_results = {
                'actionType': 'getNearest',
                'sessionId': session_id,
                'data': {
                    'images': results
                }
            }
            with open('../data/top10querylog.txt', 'a') as f:
                f.write('Result: \n{}\n\n'.format(results))
            print('[Query TOP10] Result: \n{}'.format(results))
            rabbit_mq.send_with_exchange(Config.Queues.ACTION_RESULT,
                                         session_id, json.dumps(msg_results))
            # Those cmt for querying tracker from the image ids tracker
            # query_track_id = int(image_id.split('_')[0])
            # query_embs = [cursor['embedding'] for cursor in mongodb_faceinfo.find({'track_id': query_track_id})]

            # for emb in query_embs:
            #     predict_id, _, min_dist = matcher.match(emb, return_min_dist=True)
            #     if not predict_id in predicted_dict:
            #         predicted_dict[predict_id] = []
            #     predicted_dict[predict_id].append(min_dist)
            # avg_predicted_dict = {pid: sum(predicted_dict[pid])/float(len(predicted_dict[pid]))
            #                     for pid in predicted_dict}
            # sorted_predicted_ids = sorted(avg_predicted_dict.items(), key=lambda kv: kv[1])
            # with open('../data/query_top10_log.txt') as f:
            #     f.write('Query IMAGE_ID: ' + image_id + '\n')
            #     f.write('Results: {} \n\n'.format(sorted_predicted_ids))
            # str_results = []
            # for closest_id, dist in sorted_predicted_ids:
            #     str_results.append(Config.Rabbit.INTRA_SEP.join([closest_id, str(dist)]))
            # result_msg = Config.Rabbit.INTER_SEP.join(str_results)
        else:
            time.sleep(1)


def handle_results(checking_tracker,
                   matching_result_dict,
                   imageid_to_keyid=None,
                   dump=True):
    if checking_tracker is None:
        return False
    # dump image for dashboard if the system just recognized s1
    if dump or checking_tracker.represent_image_id is None:
        dumped_images = checking_tracker.dump_images(
            mongodb_faceinfo, imageid_to_keyid=imageid_to_keyid)
        checking_tracker.represent_image_id = dumped_images[0]
    checking_tracker.send_time = time.time()
    mongodb_dashinfo.remove({'track_id': checking_tracker.track_id})
    mongodb_dashinfo.insert_one({
        'track_id':
        checking_tracker.track_id,
        'represent_image_id':
        checking_tracker.represent_image_id,
        'face_id':
        checking_tracker.face_id,
        'is_registered':
        checking_tracker.is_new_face,
        'matching_result':
        json.dumps(matching_result_dict),
        'recognized_type':
        checking_tracker.recognized_type
    })
    # if Config.Track.SEND_RECOG_API and top_info is not None:
    #     rabbit_mq.send_top_matching_results(
    #         top_info['top1_image_id'],
    #         top_info['top1_image_url'],
    #         top_info['top_matching_list'])

    if Config.SEND_QUEUE_TO_DASHBOARD and checking_tracker.send_time is not None:
        # face_id|http://210.211.119.152/images/<track_id>|image_id|send_time
        msg_image_id = checking_tracker.represent_image_id

        queue_msg = '|'.join([
            checking_tracker.face_id,
            Config.SEND_RBMQ_HTTP + '/' + str(checking_tracker.track_id) + '/',
            msg_image_id,
            str(checking_tracker.send_time)
        ])
        rabbit_mq.send(Config.Queues.LIVE_RESULT, queue_msg)


def get_frames(frame_queue, frame_reader, re_source, cam_url, queue_reader,
               lock, is_on):
    stream_time = time.time()
    while True and is_on[0]:
        if len(frame_queue) > Config.FRAME_QUEUE_SIZE:
            lock.acquire()
            frame_queue.clear()
            lock.release()
        frame = frame_reader.next_frame()
        if frame is not None:
            lock.acquire()
            frame_queue.append(frame)
            lock.release()
            stream_time = time.time()
        else:
            if time.time() - stream_time > Config.STREAM_TIMEOUT:
                print('Trying to connect to stream again...')
                if cam_url is not None:
                    frame_reader = URLFrameReader(cam_url, scale_factor=1)
                elif queue_reader is not None:
                    frame_reader = RabbitFrameReader(rabbit_mq, queue_reader)
                else:
                    print('Empty Image Source')
                    return -1


def generic_function(cam_url, queue_reader, area, face_extractor_model,
                     re_source, multi_thread):
    '''
    This is main function
    '''
    print("Generic function")
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))

    # Variables for tracking faces
    frame_counter = 0

    # Variables holding the correlation trackers and the name per faceid
    tracking_dirs = glob.glob(Config.TRACKING_DIR + '/*')
    if tracking_dirs == []:
        number_of_existing_trackers = 0
    else:
        lof_int_trackid = [
            int(tracking_dir.split('/')[-1]) for tracking_dir in tracking_dirs
        ]
        number_of_existing_trackers = max(lof_int_trackid) + 1
    imageid_to_keyid = {}
    tracker_manager = TrackerManager(
        area,
        imageid_to_keyid=imageid_to_keyid,
        current_id=number_of_existing_trackers)

    if Config.Matcher.CLEAR_SESSION:
        clear_session_folder()

    global querying_top10_image_ids_queue
    # mongodb_faceinfo.remove({})
    # reg_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE)
    # if reg_dict is not None:
    #     for fid in reg_dict:
    #         mongodb_faceinfo.insert_one({'image_id': fid, 'face_id': reg_dict[fid]})
    #     print('Saved regdict in mongodb as collection regdict')
    matcher = FaissMatcher()
    matcher.build(
        mongodb_faceinfo, imageid_to_keyid=imageid_to_keyid, use_image_id=True)
    svm_matcher = None
    if Config.Matcher.CLOSE_SET_SVM:
        svm_matcher = SVMMatcher()
        svm_matcher.build(mongodb_faceinfo)

    track_results = TrackerResultsDict()

    if Config.CALC_FPS:
        start_time = time.time()
    if cam_url is not None:
        frame_reader = URLFrameReader(cam_url, scale_factor=1)
    elif queue_reader is not None:
        frame_reader = RabbitFrameReader(rabbit_mq, queue_reader)
    elif args.anno_mode:
        frame_reader = URLFrameReader('./nothing.mp4', scale_factor=1)
    else:
        print('Empty Image Source')
        return -1
    if not args.anno_mode:
        video_out_fps, video_out_w, video_out_h, = frame_reader.get_info()
        print(video_out_fps, video_out_w, video_out_h)
        bbox = [
            int(Config.Frame.ROI_CROP[0] * video_out_w),
            int(Config.Frame.ROI_CROP[1] * video_out_h),
            int(Config.Frame.ROI_CROP[2] * video_out_w),
            int(Config.Frame.ROI_CROP[3] * video_out_h)
        ]
        # bbox = [0, 0, video_out_w, video_out_h]

    video_out = None
    if Config.Track.TRACKING_VIDEO_OUT:
        video_out = VideoHandle(time.time(), video_out_fps, int(video_out_w),
                                int(video_out_h))

    # Turn on querying top 10 from queue
    if Config.QUERY_TOP10_MODE:
        thread = threading.Thread(target=give_this_id_10_closest_ids)
        thread.daemon = True
        thread.start()

    frame_queue = []
    lock = threading.Lock()

    if multi_thread:
        is_on = [True]
        t = threading.Thread(target=(get_frames), args=(frame_queue, frame_reader, re_source, \
                                                        cam_url, queue_reader, lock, is_on, ))
        t.start()

    try:
        while True:
            ms_msg = rabbit_mq.receive_str(Config.Queues.MERGE)
            ms_flag = 'merge'
            if ms_msg is None:
                ms_msg = rabbit_mq.receive_str(Config.Queues.SPLIT)
                ms_flag = 'split'
            if ms_msg is not None:
                merge_anchor, merge_list = extract_info_from_json(ms_msg)
                while merge_list != []:
                    image_id1 = merge_list.pop()
                    split_merge_id(ms_flag, image_id1, merge_anchor, matcher,
                                   preprocessor, face_extractor,
                                   tracker_manager, mongodb_dashinfo,
                                   mongodb_faceinfo, mongodb_mslog)
                continue

            action_msg = rabbit_mq.receive_str(Config.Queues.ACTION)
            if action_msg is not None:
                return_dict = json.loads(action_msg)
                print('Receive: {}'.format(return_dict))
                if return_dict['actionType'] == 'getNearest':
                    lock.acquire()
                    querying_top10_image_ids_queue.append(return_dict['data'])
                    lock.release()
                    continue

            if args.anno_mode:
                print('Annotation Mode, waiting for new tasks ...')
                time.sleep(1)
                continue

            if multi_thread:
                if len(frame_queue) > 0:
                    lock.acquire()
                    frame = frame_queue.pop(0)
                    lock.release()
                else:
                    frame = None
            else:
                frame = frame_reader.next_frame()

            tracker_manager.update_trackers(frame)

            #do this before check_and_recognize tracker (sync local matcher vs mongodb)

            trackers_return_dict, recognized_results = update_recognition(
                self, matcher, svm_matcher, mongodb_faceinfo)
            for tracker, matching_result_dict in recognized_results:
                handle_results(tracker, matching_result_dict, imageid_to_keyid = imageid_to_keyid, \
                                                                                    dump=False)

            # trackers_return_dict = tracker_manager.find_and_process_end_track(mongodb_faceinfo)
            track_results.merge(trackers_return_dict)

            tracker_manager.long_term_history.check_time(
                matcher, mongodb_faceinfo)

            if frame is None:
                print("Waiting for the new image")
                # if Config.Track.RECOGNIZE_FULL_TRACK:
                #     overtime_track_ids = tracker_manager.find_overtime_current_trackers(
                #         time_last=Config.Track.CURRENT_EXTRACR_TIMER-5,
                #         find_unrecognized=True
                #     )

                #     for overtime_track_id in overtime_track_ids:
                #         checking_tracker, top_predicted_face_ids, matching_result_dict = \
                #             tracker_manager.check_and_recognize_tracker(
                #                 matcher,
                #                 overtime_track_id,
                #                 mongodb_faceinfo,
                #                 svm_matcher)
                #         handle_results(checking_tracker, matching_result_dict, imageid_to_keyid = imageid_to_keyid,\
                #                                                                                  dump=False)
                if re_source and not multi_thread:
                    print('Trying to connect the stream again ...')
                    if cam_url is not None:
                        frame_reader = URLFrameReader(cam_url, scale_factor=1)
                    elif queue_reader is not None:
                        frame_reader = RabbitFrameReader(
                            rabbit_mq, queue_reader)
                    else:
                        print('Empty Image Source')
                        return -1
                break

            print("Frame ID: %d" % frame_counter)
            if "_rotate" in video:
                # rotate cw
                rotation = int(video.split("_")[-1].split(".")[0])
                frame = rotate_image_90(frame, rotation)

            if Config.Track.TRACKING_VIDEO_OUT:
                video_out.tmp_video_out(frame)
            if Config.CALC_FPS:
                fps_counter = time.time()

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                # crop frame
                #frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
                origin_bbs, points = detector.detect_face(frame)
                if len(origin_bbs) > 0:
                    origin_bbs = [origin_bb[:4] for origin_bb in origin_bbs]
                    display_and_padded_faces = [
                        CropperUtils.crop_display_face(frame, origin_bb)
                        for origin_bb in origin_bbs
                    ]
                    #cropped_faces = [CropperUtils.crop_face(frame, origin_bb) for origin_bb in origin_bbs]
                    preprocessed_images = [
                        preprocessor.process(
                            CropperUtils.crop_face(frame, origin_bb))
                        for origin_bb in origin_bbs
                    ]
                    embeddings_array, _ = face_extractor.extract_features_all_at_once(
                        preprocessed_images)
                for i, origin_bb in enumerate(origin_bbs):

                    display_face, str_padded_bbox = display_and_padded_faces[i]
                    #cropped_face = CropperUtils.crop_face(frame, origin_bb)

                    # Calculate embedding
                    preprocessed_image = preprocessed_images[i]
                    # preprocessed_image = align_preprocessor.process(frame, points[:,i], aligner, 160)
                    emb_array = np.asarray([embeddings_array[i]])

                    face_info = FaceInfo(
                        #oigin_bb.tolist(),
                        #emb_array,
                        frame_counter,
                        origin_bb,
                        points[:, i]
                        #display_face,
                        #str_padded_bbox,
                        #points[:,i].tolist()
                    )

                    is_good_face = handle_filters(points[:, i], coeff_extractor,
                                                  face_info, preprocessed_image)

                    face_info.is_good = is_good_face
                    # TODO: refractor matching_detected_face_with_trackers
                    matched_track_id = tracker_manager.track(face_info)

                    if not face_info.is_good:
                        print('BAD FACE')
                        continue

                    # Update tracker_manager
                    tracker_manager.update(matched_track_id, frame, face_info)
                    checking_tracker = None
                    # if not Config.Track.RECOGNIZE_FULL_TRACK:
                    #     checking_tracker, top_predicted_face_ids, matching_result_dict = \
                    #         tracker_manager.check_and_recognize_tracker(
                    #             matcher,
                    #             matched_track_id,
                    #             mongodb_faceinfo,
                    #             svm_matcher)
                    #     handle_results(checking_tracker, matching_result_dict, imageid_to_keyid = imageid_to_keyid, \
                    #                                                                 dump=True)

            # if Config.Track.RECOGNIZE_FULL_TRACK:
            #     overtime_track_ids = tracker_manager.find_overtime_current_trackers(
            #         time_last=Config.Track.CURRENT_EXTRACR_TIMER-5,
            #         find_unrecognized=True
            #     )

            #     for overtime_track_id in overtime_track_ids:
            #         checking_tracker, top_predicted_face_ids, matching_result_dict = \
            #             tracker_manager.check_and_recognize_tracker(
            #                 matcher,
            #                 overtime_track_id,
            #                 mongodb_faceinfo,
            #                 svm_matcher)
            #         handle_results(checking_tracker, matching_result_dict, imageid_to_keyid = imageid_to_keyid, \
            #                                                                                             dump=False)

            frame_counter += 1
            if Config.CALC_FPS:
                print("FPS: %f" % (1 / (time.time() - fps_counter)))
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
        Config.Track.CURRENT_EXTRACR_TIMER = 0
        trackers_return_dict = tracker_manager.find_and_process_end_track(
            mongodb_faceinfo)
        Config.Track.HISTORY_CHECK_TIMER = 0
        Config.Track.HISTORY_EXTRACT_TIMER = 0
        tracker_manager.long_term_history.check_time(matcher, mongodb_faceinfo)
    except KeyboardInterrupt:
        if multi_thread:
            is_on[0] = False
            t.join()
        print('Keyboard Interrupt !!! Release All !!!')
        Config.Track.CURRENT_EXTRACR_TIMER = 0
        trackers_return_dict = tracker_manager.find_and_process_end_track(
            mongodb_faceinfo)
        Config.Track.HISTORY_CHECK_TIMER = 0
        Config.Track.HISTORY_EXTRACT_TIMER = 0
        tracker_manager.long_term_history.check_time(matcher, mongodb_faceinfo)
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format(
                (frame_counter + 1) / (time.time() - start_time)))
        frame_reader.release()
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
            video_out.release()


def config_logger(logger):
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(Config.DATA_DIR, 'generic_detection_tracking.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def rotate_image_90(im, angle):
    if angle % 90 == 0:
        angle = angle % 360
        if angle == 0:
            return im
        elif angle == 90:
            return im.transpose((1, 0, 2))[:, ::-1, :]
        elif angle == 180:
            return im[::-1, ::-1, :]
        elif angle == 270:
            return im.transpose((1, 0, 2))[::-1, :, :]

    else:
        raise Exception('Error')


if __name__ == '__main__':
    # Main steps
    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--cam_url', help='your camera ip address', default=None)
    parser.add_argument('-test_all', '--test_all', action='store_true')

    parser.add_argument(
        '-a',
        '--area',
        help='The area that your ip camera is recording at',
        default='None')
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
        default=Config.FACENET_DIR)
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
    parser.add_argument(
        '-anno', '--anno_mode', help='Annotation mode', action='store_true')

    parser.add_argument(
        '-mthread',
        '--multi_thread',
        help='multi_thread mode',
        action='store_true')
    args = parser.parse_args()

    # Run
    if args.video_out is not None:
        Config.Track.TRACKING_VIDEO_OUT = True
        Config.Track.VIDEO_OUT_PATH = args.video_out
    Config.SEND_QUEUE_TO_DASHBOARD = args.dashboard
    Config.Matcher.CLEAR_SESSION = args.clear_session
    Config.Track.SEND_RECOG_API = args.rethinkdb

    face_rec_graph_face = FaceGraph()
    coeff_graph = FaceGraph()
    face_extractor = FacenetExtractor(
        face_rec_graph_face, model_path=args.face_extractor_model)
    coeff_extractor = FacenetExtractor(coeff_graph, model_path=Config.COEFF_DIR)
    detector = MTCNNDetector(
        face_rec_graph_face, scale_factor=Config.MTCNN.SCALE_FACTOR)
    preprocessor = Preprocessor()
    align_preprocessor = Preprocessor(algs=align_and_crop)
    aligner = AlignCustom()

    if args.test_all:
        videos = [
            os.path.join(args.cam_url, video)
            for video in os.listdir(args.cam_url)
            if not os.path.isdir(os.path.join(args.cam_url, video))
        ]
        for video in videos:
            print(video)
            generic_function(video, args.queue_reader, args.area,
                             args.face_extractor_model, args.re_source,
                             args.multi_thread)
    else:
        generic_function(args.cam_url, args.queue_reader, args.area,
                         args.face_extractor_model, args.re_source,
                         args.multi_thread)
