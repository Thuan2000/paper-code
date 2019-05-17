import time
import cv2
import glob
import os
from config import Config
# from tracker import TrackersList
from core.cv_utils import CropperUtils, PickleUtils
from config import Config
from core.preprocess import Preprocessor
from utils.rabbitmq import RabbitMQ
import multiprocessing


def extract_images(trackers, person_id):
    print('Extract Register Images ...')
    reg_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE)
    if reg_dict is None:
        reg_dict = {}
    first_frame_id_tracked = trackers[0].elements[0].frame_id
    b_f = first_frame_id_tracked - 1
    a_f = first_frame_id_tracked + 1
    # first_time = trackers[0].elements[0].time_stamp
    the_last_fids = [
        fid for fid in trackers.keys()
        if (b_f <= trackers[fid].elements[0].frame_id <= a_f)
    ]
    # counter_dict = {fid: len(trackers[fid].elements) for fid in trackers.keys()}
    # max_len_fid = max(counter_dict, key=counter_dict.get)
    if not os.path.exists('../data/tracking/register-{}'.format(person_id)):
        os.mkdir('../data/tracking/register-{}'.format(person_id))

    for fid in the_last_fids:
        for element in trackers[fid].elements:
            # Write images out
            image_id = '{}_{}_{}_{}'.format(person_id, element.frame_id,
                                            element.element_id,
                                            element.time_stamp)
            live_tup = ((element.face_image), (element.embedding))
            PickleUtils.save_pickle(
                '{}/{}.pkl'.format(Config.LIVE_DIR, image_id), value=live_tup)
            with open('register_function_log.txt', 'a') as f:
                f.write('Save pkl {}/{}.pkl'.format(Config.LIVE_DIR, image_id))

            img_path = '../data/tracking/register-{}/{}.jpg'.format(
                person_id, image_id)
            # cv2.imwrite(img_path,
            #             cv2.cvtColor(element.face_image, cv2.COLOR_BGR2RGB))
            # Save register
            reg_dict[image_id] = person_id
    PickleUtils.save_pickle(Config.REG_IMAGE_FACE_DICT_FILE, value=reg_dict)


def sort_filenames(file_urls):
    split_url_0 = file_urls[0].split('/')
    file_url_msg = '/'.join(
        [split_url_0[i] for i in range(len(split_url_0) - 1)])
    file_ext = '.' + split_url_0[-1].split('.')[1]
    file_names = [
        int(file_name.split('/')[-1].split('.')[0]) for file_name in file_urls
    ]
    file_names = sorted(file_names)
    return [
        file_url_msg + '/' + str(file_name) + file_ext
        for file_name in file_names
    ]


def save_frames(file_url_msg):
    read_flag = 'video'
    saved_frames = []
    file_urls = glob.glob(file_url_msg + '/*.webm')
    if file_urls == []:
        read_flag = 'images'
        file_urls = glob.glob(file_url_msg + '/*.jpg')
    if file_urls == []:
        read_flag = 'images'
        file_urls = glob.glob(file_url_msg + '/*.jpeg')
    if file_urls == []:
        print('NO VALID FILES')
        with open('register_file_log.txt', 'a') as f:
            f.write(file_url_msg + '\n')
        return []
    file_urls = sort_filenames(file_urls)
    print('Saving file ...')
    if read_flag == 'video':
        for file_url in file_urls:
            cap = cv2.VideoCapture(file_url)
            while True:
                _, frame = cap.read()
                if frame is None:
                    break
                saved_frames.append(frame)
    if read_flag == 'images':
        for file_url in file_urls:
            saved_frames.append(cv2.imread(file_url))
    return saved_frames


def register_function(detector, preprocessor, face_extractor, rabbit_mq):
    '''
    dia dia
    '''
    # Doc queue
    msg_list = rabbit_mq.receive_once('{}-register'.format(Config.DEMO_FOR))
    if msg_list is None:
        return False
    person_id = msg_list[0]  # .lower()
    file_url_msg = msg_list[1]
    list_of_trackers = TrackersList()
    frame_counter = 0
    saved_frames = save_frames(file_url_msg)
    if saved_frames == []:
        return False
    print('Detecting Faces and Extracting Features ...')
    saved_frames.reverse()
    for frame in saved_frames:
        list_of_trackers.update_dlib_trackers(frame)
        origin_bbs, points = detector.detect_face(frame)
        if origin_bbs is None:
            break
        for i, origin_bb in enumerate(origin_bbs):
            display_face, padded_bbox = CropperUtils.crop_display_face(
                frame, origin_bb)
            cropped_face = CropperUtils.crop_face(frame, origin_bb)
            # Calculate embedding
            preprocessed_image = preprocessor.process(cropped_face)
            emb_array = face_extractor.extract_features(preprocessed_image)
            matched_fid = list_of_trackers.matching_face_with_trackers(
                frame, frame_counter, origin_bb, emb_array)
            # Update list_of_trackers
            list_of_trackers.update_trackers_list(
                matched_fid,
                time.time(),
                origin_bb,
                display_face,
                emb_array,
                -1,
                'register',
                frame_counter,
                padded_bbox,
                None,
                rabbit_mq,
                mode='register')
        frame_counter += 1
        if frame_counter > 601:
            break
    if list_of_trackers.trackers == {}:
        return False

    # extract images
    extract_images(list_of_trackers.trackers, person_id)
    return True


def threading_register_function(detector, face_extractor, register_queue,
                                matcher):
    '''
    dia dia
    '''
    # Doc queue
    preprocessor = Preprocessor()
    while True:
        try:
            register_msg = register_queue.get(False)
        except multiprocessing.queues.Empty:
            register_msg = None
        if register_msg:
            person_id = register_msg[0]  # .lower()
            file_url_msg = register_msg[1]
            list_of_trackers = TrackersList()
            frame_counter = 0
            saved_frames = save_frames(file_url_msg)
            if saved_frames == []:
                continue
            print('Detecting Faces and Extracting Features ...')
            saved_frames.reverse()
            for frame in saved_frames:
                list_of_trackers.update_dlib_trackers(frame)
                origin_bbs, points = detector.detect_face(frame)
                if origin_bbs is None:
                    break
                for i, origin_bb in enumerate(origin_bbs):
                    display_face, padded_bbox = CropperUtils.crop_display_face(
                        frame, origin_bb)
                    cropped_face = CropperUtils.crop_face(frame, origin_bb)
                    # Calculate embedding
                    preprocessed_image = preprocessor.process(cropped_face)
                    emb_array = face_extractor.extract_features(
                        preprocessed_image)
                    matched_fid = list_of_trackers.matching_face_with_trackers(
                        frame, frame_counter, origin_bb, emb_array)
                    # Update list_of_trackers
                    list_of_trackers.update_trackers_list(
                        matched_fid,
                        time.time(),
                        origin_bb,
                        display_face,
                        emb_array,
                        -1,
                        'register',
                        frame_counter,
                        padded_bbox,
                        None,
                        rabbit_mq,
                        mode='register')
                frame_counter += 1
                if frame_counter > 601:
                    break
            if list_of_trackers.trackers == {}:
                return False

            # extract images
            extract_images(list_of_trackers.trackers, person_id)
            matcher.update(force=True)
            with open('register_log.txt', 'a') as f:
                f.write(str(len(list_of_trackers.trackers)) + person_id + '\n')


def clear_register(rabbit_mq):
    msg_list = rabbit_mq.receive_once('{}-command'.format(Config.DEMO_FOR))
    if msg_list is None:
        return False
    command_msg = msg_list[0] #.lower()
    return command_msg
