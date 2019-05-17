import os
import time
from cv_utils import create_if_not_exist, PickleUtils, CropperUtils
import cv2
from random import shuffle
from scipy import misc
from rabbitmq import RabbitMQ
from pymongo import MongoClient
import argparse
from config import Config
import glob
from shutil import copyfile
#rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
#(Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)

mongodb_db = mongodb_client["eyeq-vingroup-production"]
mongodb_db1 = mongodb_client["vinrecognition"]
mongodb_faceinfo = mongodb_db1[Config.MongoDB.FACEINFO_COLS_NAME]


def mongo_benchmark():
    cursors = mongodb_faceinfo.find({})
    image_id_list = [cursor["_id"] for cursor in cursors]
    shuffle(image_id_list)

    N = 100
    count = 0
    total = 0
    i = 0
    while i < len(image_id_list):
        count += 1
        start = time.time()
        for c in range(i, min(len(image_id_list), i + N)):
            mongodb_faceinfo.find_one({"_id": image_id_list[c]})
        total += time.time() - start
        print(time.time() - start)
        i += N
    print("average time: " + str(total / count))


web_log = mongodb_db["visithistory"]

track_folder = "../data/tracking"
replicate_folder = track_folder + "_"
create_if_not_exist(replicate_folder)
cursors = web_log.find({})
for cursor in cursors:
    image_info = cursor["image"].split("/")[-2:]
    track_id = image_info[0]
    image_name = image_info[1]
    image_path = os.path.join(track_folder, track_id, image_name)
    create_if_not_exist(os.path.join(replicate_folder, track_id))
    copyfile(image_path, os.path.join(replicate_folder, track_id, image_name))
