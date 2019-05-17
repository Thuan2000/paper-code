import os
import glob
import time
import imageio
import numpy as np
import sys
import cv2
from time import sleep
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from config import Config
from matcher import FaissMatcher
from tracker import TrackersList, TrackerResultsDict
from preprocess import Preprocessor
from cv_utils import (CropperUtils, clear_tracking_folder, clear_session_folder)


def create_fake_frame(detection):
    image = imageio.imread(detection[1])
    full_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    ori = detection[2]
    pad = detection[3]
    top_left_x = ori[0] - pad[0]
    top_left_y = ori[1] - pad[1]
    w = image.shape[1]
    h = image.shape[0]
    full_image[top_left_y - 16: top_left_y + h - 16, top_left_x - 16: top_left_x + w - 16, :] \
        = image
    return full_image


def gen_images_with_time(root_folder):
    dir_list = []
    for folder in os.listdir(root_folder):
        dir_list.append(folder)
    images_with_time = []
    for dir in dir_list:
        for file in glob.glob(os.path.join(root_folder, dir, '*')):
            file_name = os.path.split(file)[-1]
            frame_id = file_name.split('_')[5]
            origin_bbox = [int(i) for i in file_name.split('_')[1:5]]
            padding_bbox = [
                int(i) for i in file_name.split('.')[1].split('_')[-4:]
            ]
            images_with_time.append((int(frame_id), file, origin_bbox,
                                     padding_bbox))
    return sorted(images_with_time)


class FakeMQ(object):

    def __init__(self):
        pass


def simulate_tracking(root_folder):
    Config.Track.FACE_TRACK_IMAGES_OUT = True
    Config.Track.SEND_FIRST_STEP_RECOG_API = False
    Config.Track.MIN_MATCH_DISTACE_OUT = True
    Config.Track.CURRENT_EXTRACR_TIMER = 5

    # Load Face Extractor
    face_rec_graph = FaceGraph()
    face_rec_graph_coeff = FaceGraph()
    face_extractor = FacenetExtractor(
        face_rec_graph, model_path=Config.FACENET_DIR)
    coeff_extractor = FacenetExtractor(
        face_rec_graph_coeff, model_path=Config.COEFF_DIR)

    # Create empty KDTreeMatcher
    matcher = FaissMatcher()
    matcher._match_case = 'TCH'

    # Preprocessor
    preprocessor = Preprocessor()

    # Fake rabbit mq

    rabbit_mq = FakeMQ()

    # Clean up for
    clear_tracking_folder()
    if Config.Matcher.CLEAR_SESSION:
        clear_session_folder()

    # Setup result list
    list_of_trackers = TrackersList()
    track_results = TrackerResultsDict()
    predict_dict = {}
    confirmed_ids_dict = {}

    sim_detections = gen_images_with_time(root_folder)
    for detection in sim_detections:
        frame = create_fake_frame(detection)
        sleep(0.05)
        trackers_return_dict, predict_trackers_dict = \
            list_of_trackers.check_delete_trackers(matcher, rabbit_mq)
        track_results.update_two_dict(trackers_return_dict)
        predict_dict.update(predict_trackers_dict)
        confirmed_ids_dict = list_of_trackers.trackers_history.confirm_id(
            confirmed_ids_dict)
        list_of_trackers.trackers_history.check_time(matcher)
        list_of_trackers.update_dlib_trackers(frame)
        facial_quality = 1
        # Crop face for features extraction

        origin_bb = detection[2]
        display_face, padded_bbox = CropperUtils.crop_display_face(
            frame, origin_bb)
        cropped_face = CropperUtils.crop_face(frame, origin_bb)

        bbox_str = '_'.join(np.array(origin_bb, dtype=np.unicode).tolist())
        # Calculate embedding
        preprocessed_image = preprocessor.process(cropped_face)
        emb_array, _ = face_extractor.extract_features(preprocessed_image)
        _, coeff = coeff_extractor.extract_features(preprocessed_image)
        if coeff < 0.15:
            img_path = '../data/notenoughcoeff/{}_{}_{}.jpg'.format(
                detection, bbox_str, coeff)
            cv2.imwrite(img_path, cv2.cvtColor(display_face, cv2.COLOR_BGR2RGB))
            facial_quality = -1
        else:
            with open('../data/coeff_log.txt', 'a') as f:
                f.write('{}_{}_{}, coeff: {}\n'.format(bbox_str, detection[0],
                                                       padded_bbox, coeff))

        matched_fid = list_of_trackers.matching_face_with_trackers(
            frame, detection[0], origin_bb, emb_array, facial_quality)
        if facial_quality == -1 or coeff < 0.15:
            continue

        list_of_trackers.update_trackers_list(
            matched_fid, time.time(), origin_bb, display_face, emb_array, 0,
            'VVT', 1, detection[0], padded_bbox, matcher, rabbit_mq)

        list_of_trackers.check_recognize_tracker(matcher, rabbit_mq,
                                                 matched_fid)

    sleep(6)
    list_of_trackers.check_delete_trackers(matcher, rabbit_mq)


simulate_tracking(sys.argv[1])
