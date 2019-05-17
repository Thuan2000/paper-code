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
import time
import os
import cv2
from scipy import misc
from matcher import KdTreeMatcher
from config import Config
from preprocess import Preprocessor
from web_register import clear_register, save_frames
from cv_utils import (FaceAngleUtils, CropperUtils, clear_tracking_folder,
                      is_inner_of_range, clear_session_folder, PickleUtils,
                      calc_bb_percentage, draw_img)
from face_detector import MTCNNDetector
from tf_graph import FaceGraph
import glob
from tracker_manager import TrackerManager
from tracking_utils import FaceInfo
import queue


def clear_person_id_in_reg(reg_dict, person_id):
    # Clear pickles
    delete_live_pkl_dirs = glob.glob(
        '../session/live/{}_*.pkl'.format(person_id))
    for pkl_dir in delete_live_pkl_dirs:
        os.remove(pkl_dir)

    # Clear in register dictionary and return
    return {
        image_id: reg_dict[image_id]
        for image_id in reg_dict
        if reg_dict[image_id] != person_id
    }


def extract_images(trackers, person_id):
    print('Extract Register Images ...')
    reg_dict = PickleUtils.read_pickle(Config.REG_IMAGE_FACE_DICT_FILE)
    if reg_dict is None:
        reg_dict = {}

    # Clear reg_dict and pickles in live folder
    reg_dict = clear_person_id_in_reg(reg_dict, person_id)
    print('Tracker len {}'.format(len(trackers)))
    print('Tracker ids')
    for i in trackers.keys():
        print(i)
    first_frame_id_tracked = trackers[0].elements[0].frame_id
    b_f = first_frame_id_tracked - 5
    a_f = first_frame_id_tracked + 5
    the_last_fids = [
        fid for fid in trackers.keys()
        if (b_f <= trackers[fid].elements[0].frame_id <= a_f)
    ]
    # counter_dict = {fid: len(trackers[fid].elements) for fid in trackers.keys()}
    # max_len_fid = max(counter_dict, key=counter_dict.get)
    if not os.path.exists('../data/tracking/register-{}'.format(person_id)):
        os.mkdir('../data/tracking/register-{}'.format(person_id))
    face_embs = []
    face_labels = []
    if len(the_last_fids) > 1:
        return face_embs, face_labels, 'many_faces'
    for fid in the_last_fids:
        if len(trackers[fid].elements) < 100:
            return face_embs, face_labels, 'not_good'
        with open('register_function_log.txt', 'a') as f:
            f.write('Len of elements {}\n'.format(len(trackers[fid].elements)))
        for index, element in enumerate(trackers[fid].elements):
            face_embs.append(element.embedding)

            # Write images out

            image_id = '{}_{}_{}_{}'.format(person_id, element.frame_id, index,
                                            element.time_stamp)
            live_tup = ((element.display_image), (element.embedding))
            PickleUtils.save_pickle(
                '{}/{}.pkl'.format(Config.LIVE_DIR, image_id), value=live_tup)
            with open('register_function_log.txt', 'a') as f:
                f.write('Save pkl {}/{}.pkl'.format(Config.LIVE_DIR, image_id))
            face_embs.append(element.embedding)
            # if not os.path.exists('../data/register/register-{}'.format(person_id)):
            #     os.mkdir('../data/register/register-{}'.format(person_id))
            # img_path = '../data/register/register-{}/{}.jpg'.format(person_id,
            #                                                         image_id)
            # misc.imsave(img_path, element.display_image)
            # Save register
            reg_dict[image_id] = person_id
    PickleUtils.save_pickle(Config.REG_IMAGE_FACE_DICT_FILE, value=reg_dict)
    face_labels = [person_id] * len(face_embs)
    return face_embs, face_labels, 'ok'


def generic_function(frame_readers, area, session_id, detector, face_extractor,
                     matcher, register_commands, sent_msg_queue):
    '''
    This is main function
    '''
    print("Area: {}".format(area))
    print('Thread {} created'.format(session_id))
    frame_counter = 0
    tracker_manager = TrackerManager(area)

    # clear_tracking_folder()

    # if Config.Matcher.CLEAR_SESSION:
    #     clear_session_folder()

    if not os.path.exists(Config.SEND_RBMQ_DIR):
        os.mkdir(Config.SEND_RBMQ_DIR)

    preprocessor = Preprocessor()
    # matcher = KdTreeMatcher()
    # matcher._match_case = 'TCH'
    # face_extractor = components['face_ext']
    # detector = components['detector']
    # face_rec_graph = FaceGraph()
    # detector = MTCNNDetector(face_rec_graph)

    # face_cascade = components['face_cascade']
    # eye_detector = components['eye_detector']
    # mouth_detector = components['mouth_detector']

    frame_reader = frame_readers[session_id]
    register_command = register_commands[session_id]
    if Config.CALC_FPS:
        start_time = time.time()
    unavailable_counter = time.time()
    last_labels = 'empty'
    # matcher.build(Config.REG_IMAGE_FACE_DICT_FILE)
    try:
        while True:
            try:
                reg_msg_list = register_command.get(False)
            except queue.Empty:
                reg_msg_list = None
            if reg_msg_list is not None:
                print(reg_msg_list)
                update_message = '{}|register_ko|Register Fail'.format(
                    session_id)
                person_id = reg_msg_list[0]  # .lower()
                file_url_msg = reg_msg_list[1]
                list_of_reg_trackers = TrackerManager(area)
                frame_counter = 0
                saved_frames = save_frames(file_url_msg)
                if saved_frames == []:
                    print("save frames is None")
                    update_message = '{}|register_ko|Empty Source or Invalid Format'.format(
                        session_id)
                else:
                    print('Detecting Faces and Extracting Features ...')
                    saved_frames.reverse()
                    for frame in saved_frames:
                        list_of_reg_trackers.update_dlib_trackers(frame)
                        origin_bbs, points = detector.detect_face(frame)
                        if origin_bbs is None:
                            print('not detect face on frame')
                            break
                        for i, origin_bb in enumerate(origin_bbs):
                            if is_inner_of_range(origin_bb, frame.shape):
                                continue
                            display_face, str_padded_bbox = CropperUtils.crop_display_face(
                                frame, origin_bb)
                            cropped_face = CropperUtils.crop_face(
                                frame, origin_bb)
                            # Calculate embedding
                            preprocessed_image = preprocessor.process(
                                cropped_face)
                            emb_array, _ = face_extractor.extract_features(
                                preprocessed_image)

                            face_info = FaceInfo(origin_bb, emb_array,
                                                 frame_counter, display_face,
                                                 str_padded_bbox)
                            matched_track_id = list_of_reg_trackers.track(
                                face_info)
                            list_of_reg_trackers.update(matched_track_id, frame,
                                                        face_info)

                        frame_counter += 1
                        if frame_counter > 601:
                            break
                    if list_of_reg_trackers.current_trackers != {}:
                        embs, lbls, result_status = extract_images(
                            list_of_reg_trackers.current_trackers, person_id)
                        if result_status == 'ok':
                            matcher.update(embs, lbls)
                            registered_ids = set(lbls)
                            registered_msg = ', '.join(registered_ids)
                            # send message to rb
                            update_message = '{}|register_ok|Registered {}'.format(
                                session_id, registered_msg)
                            print('REGISTER DONEEEEEEEEEEEEE\n')
                        elif result_status == 'many_faces':
                            print(
                                'REGISTER ERROR: Many faces or your head turns too fast'
                            )
                            # send message to rb
                            update_message = '{}|register_ko|Many faces in the sequence'.format(
                                session_id)
                        elif result_status == 'not_good':
                            update_message = '{}|register_ko|Not enough faces registerd'.format(
                                session_id)
                        else:
                            print('REGISTER ERROR')
                            # send message to rb
                            update_message = '{}|register_ko|Register Error'.format(
                                session_id)
                    else:
                        print('No tracker found')
                        update_message = '{}|register_ko|No Face Detected'.format(
                            session_id)

                sent_msg_queue.put(('{}-status'.format(Config.DEMO_FOR),
                                    update_message))

                frame_reader.clear()

            # LIVE MODE
            frame = frame_reader.next_frame()
            if frame is None:
                if time.time(
                ) - unavailable_counter >= Config.TIME_KILL_NON_ACTIVE_PROCESS:
                    if register_commands[session_id].empty():
                        frame_readers.pop(session_id, None)
                        register_commands.pop(session_id, None)
                        return
                time.sleep(1)
                tracker_manager.find_and_process_end_track()
                # print('Waiting for new frame')
                continue

            unavailable_counter = time.time()

            print("Frame ID: %d" % frame_counter)
            fps_counter = time.time()

            tracker_manager.update_dlib_trackers(frame)
            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                # display_frame = frame
                print(Config.Frame.FRAME_INTERVAL)
                detector.detect_face(frame)
                origin_bbs, points = detector.detect_face(frame)
                for i, origin_bb in enumerate(origin_bbs):
                    bb_size = calc_bb_percentage(origin_bb, frame.shape)
                    # print(bb_size)
                    if (is_inner_of_range(origin_bb, frame.shape) and
                            calc_bb_percentage(
                                origin_bb, frame.shape) > Config.Track.BB_SIZE):
                        continue

                    display_face, str_padded_bbox = CropperUtils.crop_display_face(
                        frame, origin_bb)
                    cropped_face = CropperUtils.crop_face(frame, origin_bb)
                    print('pass Crop Utils')
                    # Calculate embedding
                    preprocessed_image = preprocessor.process(cropped_face)
                    emb_array, _ = face_extractor.extract_features(
                        preprocessed_image)
                    print('calculated embedding')
                    # TODO: refractor matching_detected_face_with_trackers
                    face_info = FaceInfo(origin_bb, emb_array, frame_counter,
                                         display_face, str_padded_bbox)
                    matched_track_id = tracker_manager.track(face_info)
                    tracker_manager.update(matched_track_id, frame, face_info)
                    tracker_manager.check_and_recognize_tracker(
                        matcher, matched_track_id, short_term_add_new=False)
                    matched_tracker = tracker_manager.current_trackers[
                        matched_track_id]
                    if matched_tracker.face_id.startswith(
                            'TCH-{}'.format(area)):
                        matched_tracker.face_id = Config.Matcher.NEW_FACE

                    print('update trackers list')
                    if tracker_manager.current_trackers[
                            matched_track_id].face_id == last_labels:
                        continue
                    last_labels = tracker_manager.current_trackers[
                        matched_track_id].face_id
                    image_id = '{}_{}_{}.jpg'.format(
                        tracker_manager.current_trackers[matched_track_id].
                        face_id, time.time(), frame_counter)
                    img_dir = os.path.join(Config.SEND_RBMQ_DIR, image_id)
                    misc.imsave(img_dir, display_face)
                    face_msg = '|'.join([
                        session_id, tracker_manager.
                        current_trackers[matched_track_id].face_id,
                        'images/' + img_dir.split('/')[-1]
                    ])
                    if not Config.Matcher.NEW_FACE in face_msg:
                        # rabbit_mq.send('{}-result'.format(Config.DEMO_FOR), face_msg)
                        sent_msg_queue.put(('{}-result'.format(Config.DEMO_FOR),
                                            face_msg))

                    if matched_tracker.face_id == Config.Matcher.NEW_FACE:
                        tracker_manager.current_trackers.pop(
                            matched_track_id, None)

                    # draw frame
                    # display_frame = draw_img(
                    #     display_frame,
                    #     origin_bb,
                    #     str(bb_size)
                    #     # track_manager.current_trackers[matched_track_id].face_id
                    #     )

            # display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            # display_frame = cv2.resize(display_frame, (1280, 720))
            # cv2.imshow("FACE TRACKING SYSTEM {}".format(session_id), display_frame)
            # key = cv2.waitKey(1)
            # if key & 0xFF == ord('q'):
            #     break
            tracker_manager.find_and_process_end_track()
            frame_counter += 1
            if Config.CALC_FPS:
                print("FPS: %f" % (1 / (time.time() - fps_counter)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt !!! Release All !!!')
        tracker_manager.long_term_history.check_time(matcher)
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format(
                (frame_counter + 1) / (time.time() - start_time)))
