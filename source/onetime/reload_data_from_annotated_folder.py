import os
from cv_utils import create_if_not_exist, PickleUtils, CropperUtils
import cv2
from scipy import misc
from tf_graph import FaceGraph
from face_align import AlignCustom
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from preprocess import Preprocessor
from matcher import FaissMatcher, KdTreeMatcher
from cv_utils import create_if_not_exist
from rabbitmq import RabbitMQ
from config import Config
from pymongo import MongoClient
import argparse
import glob
import time

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


def main(data_path):
    #wipe data
    mongodb_dashinfo.remove({})
    mongodb_faceinfo.remove({})

    tracker_paths = [
        tracker_path for tracker_path in glob.glob(data_path + '/*')
        if os.path.isdir(tracker_path)
    ]
    face_rec_graph = FaceGraph()
    face_extractor = FacenetExtractor(face_rec_graph)
    preprocessor = Preprocessor()
    #get max track id
    existing_tracking_paths = os.listdir(Config.TRACKING_DIR)
    track_id = max([int(dir) for dir in existing_tracking_paths
                   ]) + 1 if len(existing_tracking_paths) > 0 else 0
    for tracker_path in tracker_paths:  #assuming that each annotated folder is a tracker
        tracker_save_folder = os.path.join(Config.TRACKING_DIR, str(track_id))
        preprocessed_images = []
        insert_list = []
        #create fake face_id
        face_id = '{}-{}-{}'.format("Office", track_id, time.time())
        display_imgs = [
            os.path.basename(_dir)
            for _dir in glob.glob(tracker_path + '/*.jpg')
        ]

        #iterate through list of img names
        for display_img in display_imgs:
            image_id = display_img.replace(".jpg", "")
            img = misc.imread(os.path.join(tracker_path, display_img))
            #parse image data
            data_split = image_id.split('_')
            data_split[0] = str(track_id)
            image_id = '_'.join(data_split)
            bbox = data_split[1:5]
            bbox = [int(i) for i in bbox]
            padded_bbox = data_split[-4:len(data_split)]
            padded_bbox = '_'.join(padded_bbox)
            time_stamp = float(data_split[5])

            cropped_face = CropperUtils.reverse_display_face(img, padded_bbox)
            preprocessed_image = preprocessor.process(cropped_face)
            emb_array, _ = face_extractor.extract_features(preprocessed_image)

            insert_list.append({
                'track_id': track_id,
                'image_id': image_id,
                'face_id': face_id,
                'time_stamp': time_stamp,
                'bounding_box': bbox,
                'embedding': emb_array.tolist(),
                'padded_bbox': padded_bbox,
                'points': None,
                'is_registered': True
            })

            #save image to TRACKING DIR
            create_if_not_exist(tracker_save_folder)
            misc.imsave(
                os.path.join(tracker_save_folder, image_id + '.jpg'), img)

            # preprocessed_images.append(preprocessor.process(cropped_face))

            # #extract embeddings all at once for performance
            # embs_array, _ = face_extractor.extract_features_all_at_once(preprocessed_images)

            # #map embedding with its corresponding image id
            # for i in range(len(s_insert_list)):
            #     insert_list[i]['embedding'] = [embs_array[i].tolist()] #embedding is saved as (1,128)

        #insert all images at once for performance
        mongodb_faceinfo.insert(insert_list)

        #add log to dash info
        mongodb_dashinfo.remove({'track_id': track_id})
        mongodb_dashinfo.insert_one({
            'track_id':
            track_id,
            'represent_image_id':
            insert_list[0]['image_id'],
            'face_id':
            face_id,
            'is_registered':
            True
        })

        #simulate cv <---> web real-time connection
        queue_msg = '|'.join([
            face_id, Config.SEND_RBMQ_HTTP + '/' + str(track_id) + '/',
            insert_list[0]['image_id'],
            str(time.time())
        ])
        rabbit_mq.send(Config.Queues.LIVE_RESULT, queue_msg)

        track_id += 1  #increment track id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-path', '--path', help='path to annotated folder', default=None)

    args = parser.parse_args()
    main(args.path)
