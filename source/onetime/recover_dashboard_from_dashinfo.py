from rabbitmq import RabbitMQ
from config import Config
from pymongo import MongoClient

rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)
mongodb_db = mongodb_client['tchanno-cv']
mongodb_dashinfo = mongodb_db['dashinfo']
mongodb_faceinfo = mongodb_db['faceinfo']


def main():
    cursors = mongodb_dashinfo.find({}).sort('track_id', 1)
    nof_cursors = cursors.count()
    for i, cursor in enumerate(cursors):
        print('remaining: ' + str(nof_cursors - i))
        image_id = cursor['represent_image_id']
        face_id = cursor['face_id']
        splitted_image_id = image_id.split('_')
        time_stamp = splitted_image_id[5]
        track_id = int(splitted_image_id[0])

        # face_id|http://210.211.119.152/images/<track_id>|image_id|send_time
        queue_msg = '|'.join([
            face_id, Config.SEND_RBMQ_HTTP + '/' + str(track_id) + '/',
            image_id, time_stamp
        ])
        rabbit_mq.send(Config.Queues.LIVE_RESULT, queue_msg)


main()
