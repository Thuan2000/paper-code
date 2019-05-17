from config import Config
from pymongo import MongoClient
import glob
import argparse
import datetime

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)
mongodb_db = mongodb_client['dashboard']
mongodb_cols = mongodb_db['visithistory']
tracking_folder = '../data/tracking'
HTTP_URL = 'http://210.211.119.152:1111/images'


def main(min):
    anchor_ts = time.time() - int(min) * 60
    cursors = mongodb_cols.find()
    for cursor in cursors:
        if float(cursor['timestamp']) > anchor_ts:
            image_id_dir = cursor['image']
            image_id = image_id_dir.split('/')[-1].strip('.jpg')
            splitted_image_id = image_id.split('_')
            track_id = splitted_image_id[0]
            img_dirs = glob.glob(tracking_folder + '/' + str(track_id) +
                                 '/*.jpg')
            img_dir = img_dirs[0]
            exist_image_id = img_dir.split('/')[-1].strip('.jpg')
            send_url = '/'.join(HTTP_URL, track_id, exist_image_id)
            print('overwrite: ' + send_url)
            mongodb_cols.update({
                'image': image_id
            }, {'$set': {
                'image': send_url
            }},
                                multi=True)
    print('Done')


parser = argparse.ArgumentParser()
parser.add_argument('-min', '--min', help='string image_id', default=None)
args = parser.parse_args()

main(args.min)
