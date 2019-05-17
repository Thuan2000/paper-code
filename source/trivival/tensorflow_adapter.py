import tensorflow as tf
from config import Config
import numpy as np
from tf_graph import FaceGraph
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from preprocess import Preprocessor, align_and_crop
from face_align import AlignCustom
import cv2


class TensorflowAdapter(object):

    @classmethod
    def __init__(cls):
        cls.face_rec_graph_face = FaceGraph()
        cls.coeff_graph = FaceGraph()
        cls.face_extractor = FacenetExtractor(
            cls.face_rec_graph_face, model_path=Config.Model.FACENET_DIR)
        cls.coeff_extractor = FacenetExtractor(
            cls.coeff_graph, model_path=Config.Model.COEFF_DIR)
        cls.detector = MTCNNDetector(
            cls.face_rec_graph_face, scale_factor=Config.MTCNN.SCALE_FACTOR)
        cls.preprocessor = Preprocessor()
        # align_preprocessor = Preprocessor(algs=align_and_crop)
        # aligner = AlignCustom()

    @classmethod
    def detect_face(cls, frame):
        origin_bbs, points = cls.detector.detect_face(frame)
        return origin_bbs, points

    @classmethod
    def extract_emb(cls, face_img):
        processed_img = cls.process_for_extractor(face_img)
        emb, _ = cls.face_extractor.extract_features(processed_img)
        return emb

    @classmethod
    def extract_embs_all_at_once(cls, face_imgs):
        processed_imgs = [cls.process_for_extractor(i) for i in face_imgs]
        embs, _ = cls.face_extractor.extract_features_all_at_once(
            processed_imgs)
        return embs

    @classmethod
    def extract_coeff(cls, face_img):
        processed_img = cls.process_for_extractor(face_img)
        _, coeff = cls.coeff_extractor.extract_features(processed_img)
        return coeff

    @classmethod
    def extract_coeffs_all_at_once(cls, face_imgs):
        processed_imgs = [cls.process_for_extractor(i) for i in face_imgs]
        _, coeffs = cls.coeff_extractor.extract_features_all_at_once(
            processed_imgs)
        return coeffs

    @classmethod
    def process_for_extractor(cls, img):
        img = cv2.resize(img,
                         (Config.Align.IMAGE_SIZE, Config.Align.IMAGE_SIZE))
        return cls.preprocessor.process(img)
