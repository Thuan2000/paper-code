import time
from pymongo import MongoClient
from bson.objectid import ObjectId
from pipe import worker
from utils.logger import logger
import prod.unilever_alert.config as Config


class DatabaseWorker(worker.Worker):

    def __init__(self, **kwargs):
        self.database = kwargs.get('database')

    def doEventTask(self, _task):
        '''
        data contain two keys: alert_type, video_path
        '''
        data = _task.depackage()
        alert_type = data['alert_type']
        video_path = data['video_name']
        data_camel = {'type': alert_type,
                      'storagePath': video_path}
        self.database.insert_alert(**data_camel)
        self.putResult(data_camel)
        logger.info('Got new alert %s, save at %s' % (alert_type, video_path))


class Database():

    def __init__(self, camera_id):
        self.mongodb_client = MongoClient(
                Config.MongoDB.IP_ADDRESS,
                Config.MongoDB.PORT)
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_camera = self.mongodb_db[Config.MongoDB.CAMERA_COL]
        self.mongodb_alert = self.mongodb_db[Config.MongoDB.ALERT_COL]
        self.mongodb_snapshot = self.mongodb_db[Config.MongoDB.SNAPSHOT_COL]
        self.camera_id = ObjectId(camera_id)

    def insert_alert(self, **kwargs):
        kwargs['camera'] = self.camera_id
        kwargs['timestamp'] = time.time()
        self.mongodb_alert.insert_one(kwargs)

    def get_rtsp_link(self):
        doc = self.mongodb_camera.find_one({'_id': self.camera_id})
        if doc is not None:
            return doc['rtsp']
        return None

    def set_camera_status(self, status):
        self.mongodb_camera.update({'_id': self.camera_id}, {'$set': {'status': status}})

