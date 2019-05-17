from pymongo import MongoClient, errors
from config import Config
import numpy as np
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from bson.objectid import ObjectId
import os


class MongoError:

    Disconnect = 'MongoDBDisconnect'


def catch_disconnecting(func, init_func):

    def func_wrapper(*args, **kwargs):
        result = MongoError.Disconnect
        while result == MongoError.Disconnect:
            try:
                result = func(*args, **kwargs)
            except errors.AutoReconnect:
                try:
                    init_func()
                except:
                    pass
                print('Caught disconnecting')
                time.sleep(5)
        return result

    return func_wrapper


def catch_disconnect_all_class_methods(Cls):

    class ClassWrapper(object):

        def __init__(self, *args, **kwargs):
            self.oInstance = Cls(*args, **kwargs)

        def __getattribute__(self, s):
            """
            this is called whenever any attribute of a NewCls object is accessed. This function first tries to
            get the attribute off NewCls. If it fails then it tries to fetch the attribute from self.oInstance (an
            instance of the decorated class). If it manages to fetch the attribute from self.oInstance, and
            the attribute is an instance method then `time_this` is applied.
            """
            try:
                x = super(ClassWrapper, self).__getattribute__(s)
            except AttributeError:
                pass
            else:
                return x
            x = self.oInstance.__getattribute__(s)
            init_func = self.oInstance.__init__

            if type(x) == type(self.__init__): # it is an instance method
                return catch_disconnecting(x, init_func)                 # this is equivalent of just decorating the method with time_this
            else:
                return x

    return ClassWrapper


class AbstractDatabase(metaclass=ABCMeta):

    def __init__(self, use_image_id=True):
        '''
        For communication with the database
        :param use_image_id: use image_id as labels instead of face_id
        '''
        self.use_image_id = use_image_id
        if Config.Mode.PRODUCTION:
            print('Connect to db', Config.MongoDB.IP_ADDRESS)
            self.mongodb_client = MongoClient(Config.MongoDB.IP_ADDRESS,
                                              Config.MongoDB.PORT)
        else:
            print('Connect to db', Config.MongoDB.IP_ADDRESS,
                  Config.MongoDB.USERNAME)
            self.mongodb_client = MongoClient(
                Config.MongoDB.IP_ADDRESS,
                Config.MongoDB.PORT,
                username=Config.MongoDB.USERNAME,
                password=Config.MongoDB.PASSWORD)
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def insert_new_face(self, **args):
        '''
        Insert new face to database
        '''
        pass

    def get_labels_and_embs(self):
        pass

    @abstractmethod
    def find_face_by_track_id(track_id):
        pass

    @abstractmethod
    def face_id_for_image_id(self, image_id):
        pass

    @abstractmethod
    def face_id_for_key_id(self, key_id):
        pass


@catch_disconnect_all_class_methods
class DashboardDatabase(AbstractDatabase):
    # TODO: Refactor

    def setup(self):
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_dashinfo = self.mongodb_db[Config.MongoDB.
                                                DASHINFO_COLS_NAME]
        self.mongodb_faceinfo = self.mongodb_db[Config.MongoDB.
                                                FACEINFO_COLS_NAME]
        self.mongodb_mslog = self.mongodb_db[Config.MongoDB.MSLOG_COLS_NAME]
        print(Config.MongoDB.FACEINFO_COLS_NAME)

    def insert_new_face(self, **args):
        self.mongodb_faceinfo.remove({'image_id': args.get('image_id')})
        self.mongodb_faceinfo.insert_one({
            'track_id':
            args.get('track_id'),
            'face_id':
            args.get('face_id'),
            'image_id':
            args.get('image_id'),
            'time_stamp':
            args.get('time_stamp'),
            'bounding_box':
            args.get('bounding_box'),
            'embedding':
            args.get('embedding'),
            'points':
            args.get('points'),
            'is_registered':
            args.get('is_registered'),
            'represent_image_id':
            args.get('represent_image_id')
        })

    def get_labels_and_embs(self):
        '''
        Get  images and labels for building matcher
        '''
        reg_image_face_dict = self.mongodb_faceinfo.find(
            {
                'is_registered': False
            }, projection={'_id': False})
        labels = []
        embs = []

        for cursor in reg_image_face_dict:
            emb = np.array(cursor['embedding'])
            embs.append(emb)
            if self.use_image_id:
                labels.append(cursor['image_id'])
            else:
                labels.append(cursor['face_id'])
        return np.array(labels), np.array(embs).squeeze()

    def push_to_dashboard(self, checking_tracker):
        '''
        Push face to dashboard for showing to user
        '''
        self.mongodb_dashinfo.remove({'track_id': checking_tracker.track_id})
        self.mongodb_dashinfo.insert_one({
            'track_id':
            checking_tracker.track_id,
            'represent_image_id':
            checking_tracker.represent_image_id,
            'face_id':
            checking_tracker.face_id,
            'is_registered':
            checking_tracker.is_new_face
        })

    def find_face_by_track_id(self, track_id):
        cursors = self.mongodb_faceinfo.find({'track_id': track_id})
        embs = []
        image_ids = []
        face_ids = []
        is_registered = False
        if cursors.count() != 0:
            is_registered = cursors[0]['is_registered']
            for cursor in cursors:
                image_ids.append(cursor['image_id'])
                embs.append(np.array(cursor['embedding']))
                face_ids.append(cursor['face_id'])
        if self.use_image_id:
            labels = image_ids
        else:
            labels = face_ids
        return image_ids, labels, embs, is_registered

    def find_face_id_by_image_id(self, image_id):
        cursor = self.mongodb_dashinfo.find({
            'represent_image_id': image_id})
        if cursor.count() > 0:
            if 'face_id' in cursor[0]:
                return cursor[0]['face_id']
        return None

    def find_face_id_by_image_id_in_faceinfo(self, image_id):
        cursor = self.mongodb_faceinfo.find({
            'image_id': image_id})
        if cursor.count() > 0:
            if 'face_id' in cursor[0]:
                return cursor[0]['face_id']
        return None

    def find_face_id_by_represent_image_id_in_faceinfo(self, represent_image_id):
        cursor = self.mongodb_faceinfo.find({
            'represent_image_id': represent_image_id})
        if cursor.count() > 0:
            if 'face_id' in cursor[0]:
                return cursor[0]['face_id']
        return None

    def face_id_for_image_id(self, image_id):
        return self.mongodb_faceinfo.find_one({'image_id': image_id})['face_id']

    def face_id_for_key_id(self, key_id):
        return self.mongodb_faceinfo.find_one({'_id': key_id})['face_id']

    def merge_split_log(self, action_type, image_id, old_face_id, new_face_id):
        self.mongodb_mslog.insert_one({
            'time_stamp': time.time(),
            'action_type': action_type,
            'image_id': image_id,
            'old_face_id': old_face_id,
            'new_face_id': new_face_id
        })

    def update_face_id(self, image_ids, track_id, new_face_id):
        self.mongodb_dashinfo.update({
            'track_id': track_id
        }, {'$set': {
            'face_id': new_face_id,
            'is_registered': False
        }},
                                     multi=True)
        for image_id in image_ids:
            self.mongodb_faceinfo.update({
                'image_id': image_id
            }, {'$set': {
                'face_id': new_face_id,
                'is_registered': False
            }},
                                         multi=True)

    def get_labels_and_embs_dashboard(self, registered_labels=[]):
        # register labels assumpt to be a list, not np array
        # return also a list, not np array like in matcher
        # TODO: Fix this
        embs = []
        labels = []
        cursors = self.mongodb_dashinfo.find({})
        nof_registered_image_ids = len(registered_labels)
        if nof_registered_image_ids < cursors.count():
            cursors = self.mongodb_dashinfo.find({
                'represent_image_id': {
                    '$nin': registered_labels
                }
            })
            unique_labels = [cursor['represent_image_id'] for cursor in cursors]
            cursors = self.mongodb_faceinfo.find({
                'image_id': {
                    '$in': unique_labels
                }
            })
            for cursor in cursors:
                embs.append(np.array(cursor['embedding']))
                labels.append(cursor['image_id'])
        return labels, embs

    def reset_collection(self):
        self.mongodb_dashinfo.drop()
        self.mongodb_mslog.drop()


@catch_disconnect_all_class_methods
class AnnotationDatabase(AbstractDatabase):

    def setup(self):
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_dataset = self.mongodb_db[Config.MongoDB.DATASET_COLS_NAME]
        self.mongodb_image = self.mongodb_db[Config.MongoDB.IMAGE_COLS_NAME]

    def get_unprocess_dataset(self):
        # TODO: change name
        cursors = self.mongodb_dataset.find({'status': 'not-ready'})
        storage_urls = []
        dataset_ids = []
        for cursor in cursors:
            storage_urls.append(cursor['storage_url'])
            dataset_ids.append(cursor['dataset_id'])
        return storage_urls, dataset_ids

    def current_processing_dataset(self, dataset_id):
        self.current_dataset_id = dataset_id

    def processed_dataset(self, dataset_id):
        self.mongodb_dataset.update({
            'dataset_id': dataset_id
        }, {'$set': {
            'status': 'ready'
        }})

    def insert_new_face(self, **args):
        #TODO: insert new face to database
        print('Insert to annotion ', '=' * 30)
        self.mongodb_image.remove({'imageId': args.get('image_id')})
        image_id = args.get('image_id')
        image = '%s/%s.jpg' % (self.current_dataset_id, image_id)
        exportImage = '%s/%s.pkl' % (self.current_dataset_id, image_id)
        time_stamp = args.get('time_stamp')
        self.mongodb_image.insert_one({
            'trackId': args.get('track_id'),
            'imageId': image_id,
            'faceId': args.get('face_id'),
            'createdAt': datetime.fromtimestamp(time_stamp),
            'timestamp': datetime.fromtimestamp(time_stamp),
            'updatedAt': datetime.now(),
            'embedding': args.get('embedding'),
            'dataset': ObjectId(self.current_dataset_id),
            'image': image,
            'exportImage': exportImage,
            'frameId': args.get('frame_id'),
            'boundingBox': args.get('bounding_box'),
            'isRemoved': False,
            'nearestImageIds': []
        })

    def find_face_by_track_id(self, track_id):
        pass

    def face_id_for_image_id(self, image_id):
        return self.mongodb_image.find_one({'imageId': image_id, 'dataset': ObjectId(self.current_dataset_id)})['faceId']

    def face_ids_for_image_ids(self, image_ids):
        cursors =  self.mongodb_image.find({'imageId': {'$in': image_ids}, 'dataset': ObjectId(self.current_dataset_id)}, {'faceId':1})
        face_ids = [c['faceId'] for c in cursors]
        return face_ids

    def image_id_for_face_id(self, face_id):
        return self.mongodb_image.find_one({'faceId': face_id, 'dataset': ObjectId(self.current_dataset_id)})['imageId']

    def face_id_for_key_id(self, key_id):
        _id = ObjectId(key_id)
        return self.mongodb_image.find_one({'_id': _id, 'dataset': ObjectId(self.current_dataset_id)})['faceId']

    def nearest_image_ids_for_image_id(self, image_id):
        return self.mongodb_image.find_one({'imageId': image_id, 'dataset': ObjectId(self.current_dataset_id)})['nearestImageIds']

    def get_labels_and_embs(self):
        images = self.mongodb_image.find({'dataset': ObjectId(self.current_dataset_id)})
        labels = []
        embs = []

        for image in images:
            embs.append(np.array(image['embedding']))
            labels.append(image['imageId'])
        return np.array(labels), np.array(embs)

    def get_ids_and_embs(self):
        images = self.mongodb_image.find({'dataset': ObjectId(self.current_dataset_id)})
        _ids = []
        embs = []

        for image in images:
            embs.append(np.array(image['embedding']))
            _ids.append(image['_id'])
        return _ids, np.array(embs)

    def get_image_ids_embs_face_ids(self):
        images = self.mongodb_image.find({'dataset': ObjectId(self.current_dataset_id)})
        image_ids = []
        embs = []
        face_ids = []

        for image in images:
            image_ids.append(image['imageId'])
            embs.append(np.array(image['embedding']))
            face_ids.append(image['faceId'])
        return np.array(image_ids), np.array(embs), np.array(face_ids)

    def get_image_records(self):
        return self.mongodb_image.find({'dataset': ObjectId(self.current_dataset_id)},
            {'imageId':1, 'faceId':1, 'embedding':1})

    def image_id_update_field(self, image_id, **kwargs):
        self.mongodb_image.update({'imageId': image_id, 'dataset': ObjectId(self.current_dataset_id)},
                            {"$set": kwargs})

    def update_field_by_ObjectId(self, _id, **kwargs):
        self.mongodb_image.update({'_id': _id}, {"$set": kwargs})

    def get_dataset_status(self):
        doc = self.mongodb_dataset.find_one({'_id': ObjectId(self.current_dataset_id)})
        if doc is not None:
            return doc['status']
        return None

    def find_face_id_by_image_id_in_faceinfo(self, image_id):
        doc = self.mongodb_image.find_one({'image_id': image_id, '_id': ObjectId(self.current_dataset_id)})
        if doc is not None:
            if 'face_id' in doc:
                return doc['face_id']
        return None


@catch_disconnect_all_class_methods
class DemoDatabase(AbstractDatabase):

    def setup(self):
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_face = self.mongodb_db[Config.MongoDB.FACE_COLS_NAME]
        self.mongodb_subcription = self.mongodb_db[Config.MongoDB.
                                                   SUBCRIPTION_COLS_NAME]

    def insert_new_face(self, **args):
        face = args.get('face')
        embs = args.get('embs').tolist(
        )  # TODO: Move tolist() inside database method for all other database
        #thumbnail = args.get('thumbnail')
        print('=' * 10, 'Insert', face)
        self.mongodb_face.update({
            '_id': ObjectId(face['id'])
        }, {"$set": {
            'embeddings': embs
        }})

    def find_face_by_track_id(self, track_id):
        pass

    def face_id_for_image_id(self, image_id):
        pass

    def face_id_for_key_id(self, key_id):
        pass

    def get_labels_and_embs(self):
        labels = None
        embs = None

        # only get face of a user
        container_id = os.environ.get('CV_SERVER_NAME')
        user_id = self.mongodb_subcription.find_one({
            'dockerContainerId':
            container_id
        })['owner']
        faces = self.mongodb_face.find({'user': user_id})
        for face in faces:
            new_embs = np.array(face['embeddings'])
            new_labels = np.array([face['name']] * len(new_embs))

            if new_embs.shape[0] == 0:
                # skip no embedding record
                continue

            if embs is None:
                embs = new_embs
                labels = new_labels
            else:
                embs = np.vstack((embs, new_embs))
                labels = np.concatenate((labels, new_labels))
        return labels, embs


@catch_disconnect_all_class_methods
class EcommerceDatabase(AbstractDatabase):

    def setup(self):
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_face = self.mongodb_db[Config.MongoDB.FACE_COLS_NAME]
        self.mongodb_stat = self.mongodb_db[Config.MongoDB.STATISTIC_COLS]

    # TODO: Handle insert new face
    def insert_new_embeddings(self, **args):
        _id = self.mongodb_face.insert({'embeddings': args.get('embeddings')})
        return _id

    def insert_new_stat(self, **args):
        self.mongodb_stat.insert_one({
            'status': args.get('status'),
            'faceId': args.get('faceId'),
            'actionType': args.get('actionType'),
            'initTimestamp': args.get('initTimestamp'),
            'receiveTimestamp': args.get('receiveTimestamp'),
            'startProcessTimestamp': args.get('startProcessTimestamp'),
            'endProcessTimestamp': args.get('endProcessTimestamp')
        })

    def insert_new_face(self, **args):
        pass

    def find_face_by_track_id(self, track_id):
        pass

    def face_id_for_image_id(self, image_id):
        pass

    def face_id_for_key_id(self, key_id):
        pass

    # TODO: Handle get labels and embs
    def get_labels_and_embs(self):
        labels = None
        embs = None

        # only get face of a user
        faces = self.mongodb_face.find()
        for face in faces:
            new_embs = np.array(face['embeddings'])
            _id = str(face['_id'])
            new_labels = np.array([_id] * len(new_embs))

            if new_embs.shape[0] == 0:
                # skip no embedding record
                continue

            if embs is None:
                embs = new_embs
                labels = new_labels
            else:
                embs = np.vstack((embs, new_embs))
                labels = np.concatenate((labels, new_labels))
        return labels, embs


@catch_disconnect_all_class_methods
class ATMAuthenticationDatabase(AbstractDatabase):
    def setup(self):
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_face = self.mongodb_db[Config.MongoDB.FACE_COLS_NAME]
        self.mongodb_stat = self.mongodb_db[Config.MongoDB.STATISTICS_COLS_NAME]
        self.mongodb_user = self.mongodb_db[Config.MongoDB.USER_COLS_NAME]

    # TODO: Handle insert new face
    def insert_new_embeddings(self, **args):
        _id = self.mongodb_face.insert({'embeddings': args.get('embeddings')})
        return _id

    def insert_new_face(self, **args):
        pass

    def find_face_by_track_id(self, track_id):
        pass

    def face_id_for_image_id(self, image_id):
        pass

    def face_id_for_key_id(self, key_id):
        pass

    # TODO: Handle get labels and embs
    def get_labels_and_embs(self):
        labels = None
        embs = None

        # only get face of a user
        faces = self.mongodb_face.find()
        for face in faces:
            new_embs = np.array(face['embeddings'])
            label = str(face['aliasId'])
            new_labels = np.array([label] * len(new_embs))

            if new_embs.shape[0] == 0:
                # skip no embedding record
                continue

            if embs is None:
                embs = new_embs
                labels = new_labels
            else:
                embs = np.vstack((embs, new_embs))
                labels = np.concatenate((labels, new_labels))
        return labels, embs

    def get_embs_by_face(self, aliasId):
        doc = self.mongodb_face.find_one({'aliasId': aliasId})
        if doc is not None:
            return np.array(doc['embeddings'])
        return np.array([])

    def insert_statistic(self, **args):
        self.mongodb_stat.insert_one(args)

    def get_aliasId_by_key_id(self, key_id):
        _id = ObjectId(key_id)
        doc = self.mongodb_face.find_one({'_id': _id})
        if doc is not None:
            return doc['aliasId']
        return None

    def get_faceId_by_aliasID(self, aliasId):
        doc = self.mongodb_face.find_one({'aliasId': aliasId})
        if doc is not None:
            return str(doc['_id'])
        return None

    def is_existed_aliasID(self, aliasId):
        return self.mongodb_face.find({'aliasId': aliasId}).count() > 0

    def get_user_info(self, aliasId):
        doc = self.mongodb_user.find_one({'primaryKey': aliasId})
        if doc is not None:
            try:
                info = doc['info']
                return info['name'], info['phone'], info['email']
            except:
                return None
        return None


@catch_disconnect_all_class_methods
class RetentionDashboardDatabase(AbstractDatabase):
    # TODO: Refactor

    def setup(self):
        self.mongodb_db = self.mongodb_client[Config.MongoDB.DB_NAME]
        self.mongodb_faceinfo = self.mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]
        self.mongodb_timestamp = self.mongodb_db[Config.MongoDB.TIME_DB_NAME]
        # dashboard_pedestrian_info for pedestrian
        # dashboard_visitor_info for face
        self.mongodb_info = self.mongodb_db[Config.MongoDB.INFO]

    def insert_new_face(self, **args):
        self.mongodb_faceinfo.remove({'imageId': args.get('imageId')})
        self.mongodb_faceinfo.insert_one({
            'trackId':
            args.get('trackId'),
            'image':
            args.get('image'),
            'exportImage':
            args.get('exportImage'),
            'frameId':
            args.get('frameId'),
            'paddedBoundingBox':
            args.get('paddedBoundingBox'),
            'faceId':
            args.get('faceId'),
            'imageId':
            args.get('imageId'),
            'timestamp':
            args.get('timestamp'),
            'boundingBox':
            args.get('boundingBox'),
            'embedding':
            args.get('embedding'),
            'points':
            args.get('points'),
            'is_registered':
            args.get('is_registered'),
            'represent_image_id':
            args.get('represent_image_id')
        })

    def get_labels_and_embs(self):
        '''
        Get  images and labels for building matcher
        '''
        reg_image_face_dict = self.mongodb_faceinfo.find(
            {
                'is_registered': False
            }, projection={'_id': False})
        labels = []
        embs = []

        for cursor in reg_image_face_dict:
            emb = np.array(cursor['embedding'])
            embs.append(emb)
            if self.use_image_id:
                labels.append(cursor['imageId'])
            else:
                labels.append(cursor['faceId'])
        return np.array(labels), np.array(embs).squeeze()

    def get_labels_and_embs_by_value_is_true_in_info(self, key):
        '''
        Get  images and labels for building matcher
        '''
        key_dict = self.mongodb_info.find(
            {
                key: True
            }, projection={'_id': False})
        labels = []
        embs = []

        for key in key_dict:
            reg_image_face_dict = self.mongodb_faceinfo.find(
            {
                'faceId': key['faceId']
            }, projection={'_id': False})

            for cursor in reg_image_face_dict:
                emb = np.array(cursor['embedding'])
                embs.append(emb)
                if self.use_image_id:
                    labels.append(cursor['imageId'])
                else:
                    labels.append(cursor['faceId'])
        return np.array(labels), np.array(embs).squeeze()

    def find_face_by_track_id(self, track_id):
        cursors = self.mongodb_faceinfo.find({'trackId': track_id})
        embs = []
        image_ids = []
        face_ids = []
        is_registered = False
        if cursors.count() != 0:
            is_registered = cursors[0]['is_registered']
            for cursor in cursors:
                image_ids.append(cursor['imageId'])
                embs.append(np.array(cursor['embedding']))
                face_ids.append(cursor['faceId'])
        if self.use_image_id:
            labels = image_ids
        else:
            labels = face_ids
        return image_ids, labels, embs, is_registered

    def find_face_id_by_image_id_in_faceinfo(self, image_id):
        cursor = self.mongodb_faceinfo.find({
            'imageId': image_id})
        if cursor.count() > 0:
            if 'faceId' in cursor[0]:
                return cursor[0]['faceId']
        return None

    def find_face_id_by_represent_image_id_in_faceinfo(self, represent_image_id):
        cursor = self.mongodb_faceinfo.find({
            'represent_image_id': represent_image_id})
        if cursor.count() > 0:
            if 'faceId' in cursor[0]:
                return cursor[0]['faceId']
        return None

    def find_time_stamp(self):
        undone = self.mongodb_timestamp.find({'isDone': False})
        done = self.mongodb_timestamp.find({'isDone': True})
        time = None
        if done.count() > 0:
            done.sort([('recordTimestamp', -1),])
            time = done[0]
            time['recordTimestamp'] = str(int(time['recordTimestamp']) + 1)
        if undone.count() > 0:
            undone.sort([('recordTimestamp', -1),])
            if int(undone[0]['recordTimestamp']) > int(time['recordTimestamp']):
                time = undone[0]
        if time != None:
            print('found last session : ', time)
            return time['recordTimestamp']
        else:
            print('last session not found')
            return '0'

    def insert_new_timestamp(self, **args):
        self.mongodb_timestamp.insert_one({
            'recordTimestamp':
            args.get('recordTimestamp'),
            'isDone':
            args.get('isDone'),
            'cameraName':
            args.get('cameraName')
        })

    def is_exist_timestamp(self, **args):
        time = self.mongodb_timestamp.find({
            'recordTimestamp':
            args.get('recordTimestamp'),
            'isDone':
            args.get('isDone'),
            'cameraName':
            args.get('cameraName')
        })
        if time.count() > 0:
            return True
        return False

    def update_done(self, recordTimestamp):
        self.mongodb_timestamp.update({
            'recordTimestamp': recordTimestamp
            }, {'$set': {
                'isDone': True
            }},
            multi=True)

    def face_id_for_image_id(self, image_id):
        return self.mongodb_faceinfo.find_one({'imageId': image_id})['faceId']

    def face_id_for_key_id(self, key_id):
        return self.mongodb_faceinfo.find_one({'_id': key_id})['faceId']

    def update_age_and_gender(self, face_id, gender, age):
        self.mongodb_info.update({"faceId": face_id},
                                 {"$set": {"age": age, "sex": gender}},
                                 upsert=True)
