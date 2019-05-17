'''
Lockers lock Lockers
'''
import os
import time
import threading
import glob
import numpy as np
import cv2
from rabbitmq import RabbitMQ
from tracking_utils import FaceInfo
from preprocess import Preprocessor
from tf_graph import FaceGraph
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from tracker_manager import TrackerManager
from matcher import FaissMatcher
from config import Config
from pymongo import MongoClient
from frame_reader import URLFrameReader
from cv_utils import create_if_not_exist, CropperUtils, FaceAngleUtils
from face_spoofing_detector import FaceSpoofingSVM, FaceSpoofingModel3
from blynk_locker import BlynkLocker


def handle_filters(point, coeff_extractor, face_info, preprocessed_image):
    is_good_face = True
    yaw_angle = FaceAngleUtils.calc_angle(point)
    pitch_angle = FaceAngleUtils.calc_face_pitch(point)
    bbox_str = '_'.join(
        np.array(face_info.bounding_box[:-1], dtype=np.unicode).tolist())

    if abs(yaw_angle) > Config.Filters.YAW:
        is_good_face = False
    if abs(pitch_angle) > Config.Filters.PITCH:
        is_good_face = False

    _, coeff_score = coeff_extractor.extract_features(preprocessed_image)
    if coeff_score < Config.Filters.COEFF:
        is_good_face = False
    return is_good_face


mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)
mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
mongodb_lockers = mongodb_db['lockers']
mongodb_lockersinfo = mongodb_db['lockersinfo']
mongodb_logs = mongodb_db['logs']

face_rec_graph = FaceGraph()
face_extractor = FacenetExtractor(face_rec_graph, model_path=Config.FACENET_DIR)
coeff_graph = FaceGraph()
coeff_extractor = FacenetExtractor(coeff_graph, model_path=Config.COEFF_DIR)
detector = MTCNNDetector(face_rec_graph)
preprocessor = Preprocessor()
spoofing_detector = FaceSpoofingModel3()
rb = RabbitMQ()
rb.channel.queue_declare(queue='{}-lockid'.format(Config.MAIN_NAME))

# Temp Config
LISTEN_FROM_QUEUE = False


def general_process(lock_id, detector, preprocessor, face_extractor,
                    blynk_locker):
    '''
    INPUT: lock_id
    '''
    # Get locker infomation
    # lock_id = 'query from mongo'
    locker_info = mongodb_lockersinfo.find({'lock_id': lock_id})[0]
    this_locker = mongodb_lockers.find({'lock_id': lock_id})[0]
    cam_url = locker_info['cam_url']
    status = this_locker['status']

    blynk_locker.processing(status)

    # init face info
    mongodb_faceinfo = mongodb_db[str(lock_id)]

    # Variables for tracking faces
    frame_counter = 0
    start_time = time.time()
    acceptable_spoofing = 0

    # Variables holding the correlation trackers and the name per faceid
    tracking_folder = os.path.join(Config.TRACKING_DIR, str(lock_id))
    create_if_not_exist(tracking_folder)
    tracking_dirs = glob.glob(tracking_folder + '/*')
    if tracking_dirs == []:
        number_of_existing_trackers = 0
    else:
        lof_int_trackid = [
            int(tracking_dir.split('/')[-1]) for tracking_dir in tracking_dirs
        ]
        number_of_existing_trackers = max(lof_int_trackid) + 1
    tracker_manager = TrackerManager(
        'LOCKID' + str(lock_id), current_id=number_of_existing_trackers)
    frame_reader = URLFrameReader(cam_url, scale_factor=1)
    matcher = FaissMatcher()

    if status == 'locked':
        embs = []
        labels = []
        cursors = mongodb_faceinfo.find({
            'face_id': this_locker['lock_face_id']
        })
        for cursor in cursors:
            embs.append(np.array(cursor['embedding']))
            labels.append(cursor['image_id'])
        nof_registered_image_ids = len(labels)
        matcher.fit(embs, labels)

    while True:
        # in case the jerk hits the button
        if time.time() - start_time > 4:
            with open('../data/locker_{}_log.txt'.format(lock_id), 'a') as f:
                f.write('[LOCKER {}] OUT OF TIME! \n\n'.format(lock_id))
            frame_reader.release()
            blynk_locker.stop_processing(status)
            return -1

        frame = frame_reader.next_frame()
        if frame is None:
            print('Invalid Video Source')
            break

        fps_counter = time.time()
        # cv2.imshow('Locker {}'.format(lock_id), frame)
        # cv2.waitKey(1)

        tracker_manager.update_trackers(frame)
        if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
            origin_bbs, points = detector.detect_face(frame)

            for i, in_origin_bb in enumerate(origin_bbs):
                origin_bb = in_origin_bb[:-1]

                display_face, str_padded_bbox = CropperUtils.crop_display_face(
                    frame, origin_bb)
                cropped_face = CropperUtils.crop_face(frame, origin_bb)

                # is_spoofing = spoofing_detector.is_face_spoofing(cropped_face)
                # if is_spoofing:
                #     acceptable_spoofing += 1
                # with open('../data/spoofing_log.txt', 'a') as f:
                #     f.write('Spoofing Detected at Locker {}: {}\n'.format(lock_id, is_spoofing))
                # if acceptable_spoofing > 5:
                #     with open('../data/locker_{}_log.txt'.format(lock_id), 'a') as f:
                #         f.write(
                #             '[LOCKER {}] STOP PROCESSING. '
                #             'SPOOFING DETECTED!\n'.format(lock_id)
                #         )
                #     frame_reader.release()
                #     blynk_locker.stop_processing(status)
                #     return -1

                # Calculate embedding
                preprocessed_image = preprocessor.process(cropped_face)
                # preprocessed_image = align_preprocessor.process(frame, points[:,i], aligner, 160)
                emb_array, _ = face_extractor.extract_features(
                    preprocessed_image)

                face_info = FaceInfo(origin_bb.tolist(), emb_array,
                                     frame_counter, display_face,
                                     str_padded_bbox, points[:, i].tolist())

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
                checking_tracker, top_predicted_face_ids, matching_result_dict = \
                    tracker_manager.check_and_recognize_tracker(
                        matcher,
                        matched_track_id,
                        mongodb_faceinfo,
                        None)
                # handle_results(checking_tracker, matching_result_dict)
                if checking_tracker is not None:
                    dumped_images = checking_tracker.dump_images(
                        mongodb_faceinfo,
                        add_new=True,
                        trackingfolder=tracking_folder)
                    checking_tracker.represent_image_id = dumped_images[0]
                    face_url = os.path.join(Config.SEND_RBMQ_HTTP, str(lock_id),
                                            str(checking_tracker.track_id),
                                            checking_tracker.represent_image_id)
                    face_url += '.jpg'
                    if status == 'available':
                        # Save locker, sign up the face
                        mongodb_lockers.remove({'lock_id': lock_id})
                        msg_dict = {
                            'lock_id': lock_id,
                            'status': 'locked',
                            'lock_face_url': face_url,
                            'lock_face_id': checking_tracker.face_id,
                            'lock_timestamp': time.time(),
                            'unlock_face_url': None,
                            'unlock_face_id': None,
                            'unlock_timestap': None
                        }
                        mongodb_lockers.insert_one(msg_dict)

                        # update logs
                        msg_dict.update({'log_timestamp': time.time()})
                        mongodb_logs.insert_one(msg_dict)
                        with open('../data/locker_{}_log.txt'.format(lock_id),
                                  'a') as f:
                            f.write(
                                '[LOCKER {}] REGISTERED FACE AS {}. LOCKED\n'.
                                format(lock_id, checking_tracker.face_id))
                        blynk_locker.stop_processing('locked')

                    elif status == 'locked':
                        # Release the locker, face verification
                        # update locker
                        msg_dict = mongodb_lockers.find(
                            {
                                'lock_id': lock_id
                            }, projection={"_id": False})[0]
                        msg_dict.update({
                            'unlock_face': face_url,
                            'unlock_timestamp': time.time()
                        })

                        if this_locker[
                                'lock_face_id'] == checking_tracker.face_id:
                            print('UNLOCK!')
                            blynk_locker.stop_processing('available')
                            mongodb_lockers.remove({'lock_id': lock_id})
                            mongodb_lockers.insert_one({
                                'lock_id': lock_id,
                                'status': 'available',
                                'lock_face_id': None,
                                'lock_face_url': None,
                                'lock_timestamp': None,
                                'unlock_face_id': None,
                                'unlock_face_url': None,
                                'unlock_timestap': None
                            })
                            with open(
                                    '../data/locker_{}_log.txt'.format(lock_id),
                                    'a') as f:
                                f.write(
                                    '[LOCKER {}] MATCHED WITH FACE ID {}. '
                                    'UNLOCKED. THIS LOCKER IS AVAILABLE NOW!\n'.
                                    format(lock_id, checking_tracker.face_id))

                        else:
                            print('NOT MATCH')
                            blynk_locker.stop_processing('locked')
                            with open(
                                    '../data/locker_{}_log.txt'.format(lock_id),
                                    'a') as f:
                                f.write('[LOCKER {}] NOT MATCH. '
                                        'PLEASE TRY AGAIN!\n'.format(lock_id))

                        # update logs
                        msg_dict.update({'log_timestamp': time.time()})
                        mongodb_logs.insert_one(msg_dict)
                    frame_reader.release()
                    return 1
            tracker_manager.find_and_process_end_track(mongodb_faceinfo)
            frame_counter += 1
            print("FPS: %f" % (1 / (time.time() - fps_counter)))


def main_process():
    # refresh lockers
    lockers_info = {}
    lockers_info = [{
        'lock_id': 0,
        'cam_url': 0,
        'token': '25573925b0a940348bdb94dec015a0a9',
    }, {
        'lock_id': 1,
        'cam_url': 1,
        'token': 'e35c1bf8b0894d979242af7187c05a83',
    }]
    mongodb_lockers.remove({})
    mongodb_lockersinfo.remove({})
    mongodb_lockersinfo.insert_many(lockers_info)

    # update lockers
    lockers_info = mongodb_lockersinfo.find({})
    lockers_dict = {}
    for locker_info in lockers_info:
        lock_id = int(locker_info['lock_id'])
        lockers_dict[lock_id] = BlynkLocker(locker_info['token'])
        with open('../data/locker_{}_log.txt'.format(lock_id), 'w') as f:
            f.write('[LOCKER {}] START ...\n\n')
        mongodb_lockers.insert_one({
            'lock_id': locker_info['lock_id'],
            'status': 'available',
            'lock_face_id': None,
            'lock_face_url': None,
            'lock_timestamp': None,
            'unlock_face_id': None,
            'unlock_face_url': None,
            'unlock_timestap': None
        })
    while True:
        # catch event
        activated_lockers = []
        if LISTEN_FROM_QUEUE:
            activated_lockers = rb.receive_once(
                queue_name='{}-lockid'.format(Config.MAIN_NAME))
            activated_lockers = [] if activated_lockers is None else activated_lockers
        else:
            for lid in lockers_dict:
                if lockers_dict[lid].is_activated():
                    activated_lockers.append(lid)

        # handle event if it does exist
        for lock_id in activated_lockers:
            if not lock_id in lockers_dict:
                print('Locker ID: {} doesnt exist!'.format(lock_id))
                continue
            if not lockers_dict[lock_id].is_processing:
                lockers_dict[lock_id].is_processing = True
                thread = threading.Thread(
                    target=general_process,
                    args=(lock_id, detector, preprocessor, face_extractor,
                          lockers_dict[lock_id]))
                thread.daemon = True
                thread.start()
            else:
                print('LOCKER {} IS RUNNING!'.format(lock_id))
        print('Waiting for the new task ...')
        time.sleep(1)


if __name__ == '__main__':
    main_process()
