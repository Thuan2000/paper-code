from shutil import copyfile, move, copy, copytree
import os
from cv_utils import create_if_not_exist, FaceAngleUtils, CropperUtils
from tf_graph import FaceGraph
import argparse
from config import Config
from face_detector import MTCNNDetector
import cv2
import time
from pymongo import MongoClient
from frame_reader import URLFrameReader, RabbitFrameReader
import subprocess, re
N = 10

face_rec_graph = FaceGraph()
detector = MTCNNDetector(face_rec_graph, scale_factor=2)

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)

mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
mongodb_dashinfo = mongodb_db[Config.MongoDB.DASHINFO_COLS_NAME]
mongodb_faceinfo = mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]


def get_bounding_box(original_path):
    restructured_path = original_path + "_restructured"
    create_if_not_exist(restructured_path)
    face_ids = [
        id for id in os.listdir(original_path)
        if os.path.isdir(os.path.join(original_path, id))
    ]
    for face_id in face_ids:
        id_path = os.path.join(original_path, face_id)
        r_id_path = os.path.join(restructured_path, face_id)
        create_if_not_exist(r_id_path)
        images = [
            os.path.join(id_path, image)
            for image in os.listdir(id_path)
            if "jpg" in image or "png" in image
        ]
        for image_name in images:
            img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
            rects, landmarks = detector.detect_face(img)
            if len(rects == 1) and FaceAngleUtils.is_acceptable_angle(
                    landmarks[:, 0]):
                origin_bb = rects[0][:4]
                display_face, str_padded_box = CropperUtils.crop_display_face(
                    img, origin_bb)
                bbox_str = '_'.join([str(int(num)) for num in origin_bb])
                image_id = '{}_{}_{}_{}.jpg'.format(face_id, bbox_str,
                                                    time.time(), str_padded_box)
                cv2.imwrite(
                    os.path.join(r_id_path, image_id),
                    cv2.cvtColor(display_face, cv2.COLOR_RGB2BGR))
                print(image_id)


def break_into_trackers(original_path):
    restructured_path = original_path + "_restructured"
    create_if_not_exist(restructured_path)

    face_ids = [
        id for id in os.listdir(original_path)
        if os.path.isdir(os.path.join(original_path, id))
    ]

    for face_id in face_ids:
        id_path = os.path.join(original_path, face_id)
        images = [
            os.path.join(id_path, image)
            for image in os.listdir(id_path)
            if "jpg" in image or "png" in image
        ]
        images.sort(key=lambda x: float(x.replace(".jpg", "").split("_")[5]))
        #images.sort(key=lambda x:int(x.split("/")[-1].split(".")[0]))
        r_id_path = os.path.join(restructured_path, face_id)
        create_if_not_exist(r_id_path)
        i = 0
        while i < len(images):
            tracker_path = os.path.join(r_id_path, str(int(i / N)))
            create_if_not_exist(tracker_path)
            for c in range(i, i + N):
                if (c < len(images)):
                    image_name = images[c].split("/")[-1]
                    copyfile(images[c], os.path.join(tracker_path, image_name))
                    print(image_name)
            i += N


def rotate_video(original_path, by_landmark=False):
    restructured_path = original_path + "_restructured"
    create_if_not_exist(restructured_path)
    videos = [os.path.join(original_path, id) for id in os.listdir(original_path)\
            if not os.path.isdir(os.path.join(original_path,id))]

    for video in videos:
        frame_reader = URLFrameReader(video, scale_factor=1)
        video_name = video.split("/")[-1].split(".")[0]
        video_type = video.split("/")[-1].split(".")[1]
        is_rotate = False
        if by_landmark:
            while True:
                frame = frame_reader.next_frame()
                if frame is None:
                    break
                rects, landmarks = detector.detect_face(frame)
                if len(rects) > 0:

                    rotate_angel = FaceAngleUtils.calc_face_rotate_angle(
                        landmarks[:, 0])
                    print("Points: " + str(landmarks[:, 0]) +
                          ", rotate_angel: " + str(rotate_angel))
                    if rotate_angel > 30:
                        video_name += "_rotate"
                    break
        else:
            cmd = 'ffmpeg -i %s' % video

            p = subprocess.Popen(
                cmd.split(" "), stderr=subprocess.PIPE, close_fds=True)
            stdout, stderr = p.communicate()
            reo_rotation = re.compile(b'rotate\s+:\s(?P<rotation>.*)')
            match_rotation = reo_rotation.search(stderr)
            if (match_rotation is not None and
                    len(match_rotation.groups()) > 0):
                rotation = match_rotation.groups()[0]

                if int(rotation) > 0:
                    video_name += "_rotate_" + str(int(rotation))

        n_video_path = os.path.join(restructured_path,
                                    video_name + "." + video_type)
        copy(video, n_video_path)
        print(video_name)


def get_trackers_from_db(track_folder):
    save_path = track_folder + "_restructured"
    create_if_not_exist(save_path)
    cursors = mongodb_faceinfo.find({})
    for cursor in cursors:
        face_folder = os.path.join(save_path, cursor["face_id"])
        try:
            copytree(
                os.path.join(track_folder, str(cursor["track_id"])),
                os.path.join(face_folder, str(cursor["track_id"])))
            print(cursor["image_id"])
        except:
            continue


def train_and_validate(original_path, train_ratio):
    restructured_path = original_path + "_restructured"
    create_if_not_exist(restructured_path)
    face_ids = [
        id for id in os.listdir(original_path)
        if os.path.isdir(os.path.join(original_path, id))
    ]
    for face_id in face_ids:
        id_path = os.path.join(original_path, face_id)
        r_id_path = os.path.join(restructured_path, face_id)
        create_if_not_exist(r_id_path)
        trackers = [
            tracker for tracker in os.listdir(id_path)
            if os.path.isdir(os.path.join(id_path, tracker))
        ]
        trackers.sort(key=lambda x: int(x))
        #create_if_not_exist(os.path.join(r_id_path,tracker))
        num_train_set = int(len(trackers) * train_ratio)
        trackers_to_move = trackers[num_train_set:]
        for t_move in trackers_to_move:
            move(os.path.join(id_path, t_move), r_id_path)


def train_validate_view_tracker(original_path):
    restructured_path = original_path + "_restructured"
    create_if_not_exist(restructured_path)
    face_ids = [
        id for id in os.listdir(original_path)
        if os.path.isdir(os.path.join(original_path, id))
    ]
    for face_id in face_ids:
        id_path = os.path.join(original_path, face_id)
        trackers = [
            tracker for tracker in os.listdir(id_path)
            if os.path.isdir(os.path.join(id_path, tracker))
        ]
        trackers.sort(key=lambda x: int(x))
        if (len(trackers) > 1):
            r_id_path = os.path.join(restructured_path, face_id)
            create_if_not_exist(r_id_path)
            for i in range(1, len(trackers)):
                move(os.path.join(id_path, trackers[i]), r_id_path)
                print(os.path.join(id_path, trackers[i]))


#train_and_validate(0.4)
#break_into_trackers()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-break',
        '--break_trackers',
        help='Break into trackers mode',
        action='store_true')

    parser.add_argument(
        '-get_bb', '--get_bb', help='Get bounding box set', action='store_true')

    parser.add_argument(
        '-path', '--path', help="path to train data", default=None)

    parser.add_argument(
        '-ratio', '--train_ratio', help='Train set ratio', default=None)

    parser.add_argument('-rotate', '--rotate', action='store_true')

    parser.add_argument(
        '-set',
        '--train_validate',
        help='create train and validate set',
        action='store_true')

    parser.add_argument(
        '-set_view',
        '--train_validate_view',
        help='create train and validate set',
        action='store_true')

    parser.add_argument(
        '-get_trackers',
        '-get_trackers',
        help='get trackers from db',
        action='store_true')

    args = parser.parse_args()
    if args.break_trackers:
        print("Break into trackers mode")
        break_into_trackers(args.path)
    elif args.train_validate:
        train_and_validate(args.path, float(args.train_ratio))
    elif args.train_validate_view:
        train_validate_view_tracker(args.path)
    elif args.get_bb:
        get_bounding_box(args.path)
    elif args.get_trackers:
        get_trackers_from_db(args.path)
    elif args.rotate:
        rotate_video(args.path)
    else:
        raise "Please choose between -break mode and -set mode"
