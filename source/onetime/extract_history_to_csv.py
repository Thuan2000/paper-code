import os
import glob
import cv2
import datetime
from cv_utils import encode_image

from config import Config
from rabbitmq import RabbitMQ

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

image_dirs = glob.glob(
    "/home/manho/source-code/iq_facial_recognition/data/send_rbmq/*.jpg")
time_stamp_list = []
info_dictionary = {}
# sort glob order by time stamp
counter = 0

for img_dir in image_dirs:
    print("Saving {}".format(img_dir))
    human_id = img_dir.split('/')[-1].split('.')[0].split('_')[0]
    img_dir_info = os.stat(img_dir)
    time_stamp = img_dir_info.st_mtime
    cv_img = cv2.imread(img_dir)
    if cv_img is None:
        print('Cant read this image')
        counter += 1
        continue
    binaryimg = encode_image(cv_img)
    time_stamp_list.append(time_stamp)
    if time_stamp in list(info_dictionary.keys()):
        raise "BREAKKKKING DOWNNNNNN"

    info_dictionary[time_stamp] = "{}|{}|{}".format(human_id, binaryimg,
                                                    time_stamp * 1000)
    print("Saved {}".format(human_id))

time_stamp_list = sorted(time_stamp_list)

f = open('/home/manho/data/tch_db_history_27062018.csv', 'a')
for fid in time_stamp_list:
    print(datetime.datetime.fromtimestamp(fid).strftime('%Y-%m-%d %H:%M:%S'))
    f.write(info_dictionary[fid] + '\n')
    # rabbit_mq.send('tchdashboard-result', info_dictionary[fid])

f.close()
print('Cant read {} images'.format(counter))
print('Done: /home/manho/data/tch_db_history.csv')
