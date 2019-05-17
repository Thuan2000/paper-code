'''
This script is for dataset folder format (face_id/track_id) to tracking folder format
and recovery data (recover dashboard, mongodb following the new format)
1. recover session to build the new matcher.
2. recover dashboard.
3. recover tracking folder.
4. update mongodb
'''

import glob
import subprocess
import os
import cv2
from scipy import misc
from rabbitmq import RabbitMQ
from config import Config
from cv_utils import PickleUtils, create_if_not_exist, CropperUtils
from pymongo import MongoClient
from face_extractor import FacenetExtractor
from preprocess import Preprocessor
from tf_graph import FaceGraph
from cv_utils import CropperUtils

facial_dirs = glob.glob('/mnt/production_data/tch_data/tch_data_Mar_June/*')
queue_name = Config.Queues.LIVE_RESULT
face_extractor_model = Config.FACENET_DIR
face_rec_graph_face = FaceGraph()
face_extractor = FacenetExtractor(
    face_rec_graph_face, model_path=face_extractor_model)
preprocessor = Preprocessor()
create_if_not_exist(Config.TRACKING_DIR)

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


def under_line_join(jlist):
    tmp = jlist
    jstr = tmp.pop(0)
    for i in tmp:
        jstr += '_' + i
    return jstr


def get_oldest_dir(track_dirs):
    '''
    return path in string
    '''
    dir_dict = {}
    if len(track_dirs) == 1:
        return track_dirs[0]
    dir_dict = {
        t_dir: float(
            glob.glob(t_dir +
                      '/*.jpg')[0].split('/')[-1].strip('.jpg').split('_')[5])
        for t_dir in track_dirs
    }
    sorted_dict = sorted(dir_dict.items(), key=lambda kv: kv[1])
    return sorted_dict[0][0]


def modify_image_id(img_path, track_id):
    img_dirs = glob.glob(img_path + '/*.jpg')
    print("Modifying name of {} files".format(len(img_dirs)))
    for img_dir in img_dirs:
        splitted_img_dir = img_dir.split('/')
        file_name = splitted_img_dir[-1].replace('.jpg', '')
        splitted_file_name = file_name.split('_')

        splitted_file_name[0] = str(track_id)
        new_file_name = under_line_join(splitted_file_name)
        ext_new_file_name = new_file_name + '.jpg'
        splitted_img_dir[-1] = ext_new_file_name
        dst_dir = '/'.join(splitted_img_dir)
        print('Modified ' + dst_dir)
        os.rename(img_dir, dst_dir)


def register_imgs(track_id_counter, is_registered, img_path, face_id,
                  mongodb_dashinfo, preprocessor, face_extractor):
    img_dirs = glob.glob(img_path + '/*.jpg')
    if img_dirs == []:
        print('ERROR: No image in ' + img_path)
        raise 'NO IMAGE IN ' + img_path
        return False
    labels = []
    embs = []

    for img_dir in img_dirs:
        image_id = img_dir.split('/')[-1].replace('.jpg', '')
        splitted_image_id = image_id.split('_')
        bbox = splitted_image_id[-4:len(splitted_image_id)]
        bbox = '_'.join(bbox)
        bounding_box = splitted_image_id[1:5]
        bounding_box = [int(bb_num) for bb_num in bounding_box]
        time_stamp = float(splitted_image_id[5])
        img = misc.imread(img_dir)
        cropped_face = CropperUtils.reverse_display_face(img, bbox)

        # Extract feature
        preprocessed_image = preprocessor.process(cropped_face)
        emb_array, _ = face_extractor.extract_features(preprocessed_image)

        # For Session
        mongodb_faceinfo.remove({'image_id': image_id})
        mongodb_faceinfo.insert_one({
            'track_id': track_id_counter,
            'image_id': image_id,
            'face_id': face_id,
            'time_stamp': time_stamp,
            'bounding_box': bounding_box,
            'embedding': emb_array.tolist(),
            'points': None,
            'is_registered': is_registered
        })


def regdict_to_faceinfo():
    mongodb_faceinfo.remove({})
    reg_dict = PickleUtils.read_pickle(reg_dict_path)
    reg_dict_length = len(reg_dict)
    for i, i_id in enumerate(reg_dict):
        print(reg_dict_length - i)
        splitted_image_id = i_id.split('_')
        track_id = int(splitted_image_id[0])
        image_id = i_id
        face_id = reg_dict[i_id]
        time_stamp = float(splitted_image_id[5])
        bounding_box = splitted_image_id[1:5]
        bounding_box = [int(bb_num) for bb_num in bounding_box]
        padded_bbox = splitted_image_id[-4:len(splitted_image_id)]
        padded_bbox = '_'.join(padded_bbox)

        display_img = PickleUtils.read_pickle(live_folder + '/' + i_id +
                                              '.pkl')[0]
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        cropped_img = CropperUtils.reverse_display_face(display_img,
                                                        padded_bbox)
        # Extract feature
        preprocessed_image = preprocessor.process(cropped_img)
        emb_array, _ = face_extractor.extract_features(preprocessed_image)

        mongodb_faceinfo.insert_one({
            'track_id': track_id,
            'image_id': image_id,
            'face_id': face_id,
            'time_stamp': time_stamp,
            'bounding_box': bounding_box,
            'embedding': emb_array.tolist(),
            'points': None,
            'is_registered': True
        })


def main():

    # print(sorted(facial_dirs))
    track_id_counter = 0
    lof_dirs = len(facial_dirs)

    for i, f_dir in enumerate(facial_dirs):
        print('Remaining: ' + str(lof_dirs - i))
        f_id = f_dir.split('/')[-1]
        track_dirs = glob.glob(f_dir + '/*')
        if f_id == 'BAD-TRACK':
            continue

        # get the registered track_dir
        registered_track_dir = get_oldest_dir(track_dirs)
        for t_dir in track_dirs:
            img_dirs = glob.glob(t_dir + '/*')
            i_dir = img_dirs[0]
            i_id = i_dir.split('/')[-1].strip('.jpg')
            time_stamp = i_id.split('_')[5]
            if float(time_stamp) < 1522540800:
                continue
            new_t_dir = os.path.join(Config.TRACKING_DIR, str(track_id_counter))

            subprocess.call(["cp", "-r", t_dir, new_t_dir])
            modify_image_id(new_t_dir, track_id_counter)

            new_i_id = glob.glob(new_t_dir +
                                 '/*.jpg')[0].split('/')[-1].strip('.jpg')

            mongodb_dashinfo.insert_one({
                'track_id':
                track_id_counter,
                'represent_image_id':
                new_i_id,
                'face_id':
                f_id,
                'is_registered':
                t_dir == registered_track_dir
            })

            queue_msg = '|'.join([
                f_id, Config.SEND_RBMQ_HTTP + '/' + str(track_id_counter) + '/',
                new_i_id, time_stamp
            ])
            rabbit_mq.send(queue_name, queue_msg)

            register_imgs(track_id_counter, t_dir == registered_track_dir,
                          new_t_dir, f_id, mongodb_dashinfo, preprocessor,
                          face_extractor)

            track_id_counter += 1


main()
