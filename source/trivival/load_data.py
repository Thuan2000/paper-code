import os
import time
from cv_utils import create_if_not_exist, PickleUtils, CropperUtils
import cv2
from scipy import misc
from tf_graph import FaceGraph
from face_align import AlignCustom
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from preprocess import Preprocessor, align_and_crop
from tracking_utils import FaceInfo, Tracker
from tracker_manager import TrackerManager
from matcher import FaissMatcher, KdTreeMatcher
from rabbitmq import RabbitMQ
from pymongo import MongoClient
import argparse
from config import Config
import glob

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)

mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
mongodb_dashinfo = mongodb_db[Config.MongoDB.DASHINFO_COLS_NAME]
mongodb_faceinfo = mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]


def main(data_path, clear):
    if (clear):
        mongodb_dashinfo.remove({})
        mongodb_faceinfo.remove({})

    tracker_paths = [
        tracker_path for tracker_path in glob.glob(data_path + '/*')
        if os.path.isdir(tracker_path)
    ]
    face_rec_graph = FaceGraph()
    face_extractor = FacenetExtractor(face_rec_graph)
    detector = MTCNNDetector(face_rec_graph, scale_factor=2)
    preprocessor = Preprocessor(algs=align_and_crop)
    aligner = AlignCustom()
    matcher = FaissMatcher()
    matcher.build(mongodb_faceinfo, use_image_id=True)
    existing_tracking_paths = os.listdir(Config.TRACKING_DIR)
    track_id = max([int(dir) for dir in existing_tracking_paths
                   ]) + 1 if len(existing_tracking_paths) > 0 else 0
    tracker_manager = TrackerManager("Office", current_id=track_id)
    for tracker_path in tracker_paths:  #assuming that each annotated folder is a tracker
        #Simulate actual realtime tracking
        display_imgs = [
            os.path.basename(_dir)
            for _dir in glob.glob(tracker_path + '/*.jpg')
        ]
        track_id += 1

        #iterate through list of img names
        for display_img in display_imgs:
            image_id = display_img.replace(".jpg", "")
            img = misc.imread(os.path.join(tracker_path, display_img))
            rects, landmarks = detector.detect_face(img)
            if len(rects) == 1:
                data_split = display_img.split('_')
                bbox = rects[0]
                padded_bbox = data_split[-4:len(data_split)]
                padded_bbox = '_'.join(padded_bbox)
                time_stamp = float(data_split[5])

                #generate embeddings
                preprocessed_face = preprocessor.process(
                    img, landmarks[:, 0], aligner, Config.Align.IMAGE_SIZE)
                embs_array, _ = face_extractor.extract_features(
                    preprocessed_face)

                #create faceinfo element
                new_element = FaceInfo(bbox.tolist(), embs_array, -1, img,
                                       padded_bbox,
                                       landmarks[:, 0].tolist())  #fake frame id
                if track_id not in tracker_manager.current_trackers:
                    tracker_manager.current_trackers[track_id] = Tracker(
                        track_id, new_element, None)
                else:
                    tracker_manager.current_trackers[track_id].elements.append(
                        new_element)
                print(display_img)

        #match tracker normally
        checking_tracker, predicted_faceid, top_info = \
             tracker_manager.check_and_recognize_tracker(matcher, track_id, mongodb_faceinfo, None)

        if checking_tracker is not None:
            dumped_images = checking_tracker.dump_images(mongodb_faceinfo)
            checking_tracker.represent_image_id = dumped_images[0]

            mongodb_dashinfo.remove({'track_id': checking_tracker.track_id})
            mongodb_dashinfo.insert_one({
                'track_id':
                checking_tracker.track_id,
                'represent_image_id':
                checking_tracker.represent_image_id,
                'face_id':
                checking_tracker.face_id,
                'is_registered':
                checking_tracker.is_new_face
            })

            #simulate cv <---> web real-time connection
            queue_msg = '|'.join([
                checking_tracker.face_id, Config.SEND_RBMQ_HTTP + '/' + str(
                    checking_tracker.track_id) + '/',
                checking_tracker.represent_image_id,
                str(time.time())
            ])
            rabbit_mq.send(Config.Queues.LIVE_RESULT, queue_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cs',
        '--clear_session',
        help='write tracking folder with good element n min distance',
        action='store_true')
    parser.add_argument(
        '-path',
        '--path',
        help='write tracking folder with good element n min distance',
        action=None)

    args = parser.parse_args()
    main(args.path, args.clear_session)
