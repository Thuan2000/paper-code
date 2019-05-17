import os
import glob
import imageio
import numpy as np
import cv2
import time
from config import Config
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from preprocess import Preprocessor
from cv_utils import CropperUtils
import pickle


def gen_frames(root_folder):
    dir_list = []
    for folder in os.listdir(root_folder):
        dir_list.append(folder)
    frames = {}
    face_rec_graph_face = FaceGraph()
    face_rec_graph_coeff = FaceGraph()
    face_extractor = FacenetExtractor(
        face_rec_graph_face, model_path=Config.FACENET_DIR)
    coeff_extractor = FacenetExtractor(
        face_rec_graph_coeff, model_path=Config.COEFF_DIR)
    preprocessor = Preprocessor()
    for dir in dir_list:
        for file in glob.glob(os.path.join(root_folder, dir, '*')):
            image = imageio.imread(file)
            identity = os.path.split(file)[-2]
            file_name = os.path.split(file)[-1]
            frame_id = file_name.split('_')[5]
            origin_bbox = [int(i) for i in file_name.split('_')[1:5]]
            if file_name.startswith('BAD-TRACK'):
                padding_bbox = [
                    int(i) for i in file_name.split('.')[0].split('_')[-4:]
                ]
            else:
                padding_bbox = [
                    int(i) for i in file_name.split('.')[1].split('_')[-4:]
                ]
            cropped_face = CropperUtils.crop_face(image, padding_bbox)
            preprocessed_image = preprocessor.process(cropped_face)
            emb_array, _ = face_extractor.extract_features(preprocessed_image)
            _, coeff = coeff_extractor.extract_features(preprocessed_image)
            if frame_id in frames:
                frames[frame_id].append((file, origin_bbox, padding_bbox,
                                         identity, emb_array))
            else:
                frames[frame_id] = [(file, origin_bbox, padding_bbox, identity,
                                     emb_array)]
    return frames


def create_fake_frame(detections):
    full_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for detection in detections:
        image = imageio.imread(detection[0])
        origin = detection[1]
        pad = detection[2]
        top_left_x = origin[0] - pad[0]
        top_left_y = origin[1] - pad[1]
        w = image.shape[1]
        h = image.shape[0]
        full_image[top_left_y - 16: top_left_y + h - 16, top_left_x - 16: top_left_x + w - 16, :] \
            = image
    return full_image


frames = gen_frames('../data/three_people')
pickle.dump(frames, open('three_people.pkl', 'wb'))

for frame_id in sorted(list(frames.keys())):
    time.sleep(0.1)
    frame = create_fake_frame(frames[frame_id])
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
