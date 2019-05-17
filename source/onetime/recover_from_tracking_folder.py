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

facial_dirs = glob.glob('tch_data_Mar_June/*')
tracking_folder_path = 'tchbk/tracking'
queue_name = 'tchbk-production'
SEND_RBMQ_HTTP = 'http://210.211.119.152:1111/tchbk_images'
face_extractor_model = Config.FACENET_DIR
face_rec_graph_face = FaceGraph()
face_extractor = FacenetExtractor(
    face_rec_graph_face, model_path=face_extractor_model)
preprocessor = Preprocessor()

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)
mongodb_db = mongodb_client['tchbk-cv']
mongodb_cols = mongodb_db['trackinginfo']
mongodb_dashinfo = mongodb_db['dashinfo']
mongodb_faceinfo = mongodb_db['faceinfo']
