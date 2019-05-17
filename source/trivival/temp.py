import time
from config import Config
from tracking.tracker_results_dict import TrackerResultsDict
from matcher import FaissMatcher, KdTreeMatcher, SVMMatcher
import pipe
import functools
import traceback
from rabbitmq import RabbitMQ
from pymongo import MongoClient
from cv_utils import PickleUtils
from timer import TimerCollector
import json
import numpy as np
import os

from threading import Thread
import threading


class FindSimilarFaceThread(Thread):
    '''
    Find similar faces from dashboard
    '''

    def __init__(self, **args):
        self.nrof_closest_faces = args.get('nrof_closest_faces', 10)
        self.database = args.get('database')
        self.rabbit_mq = args.get('rabbit_mq')
        self.event = threading.Event()
        self.setup_matcher()
        super(FindSimilarFaceThread, self).__init__()

    def join(self, timeout=None):
        print('Find similar joint')
        self.event.set()
        super(FindSimilarFaceThread, self).join()

    def setup_matcher(self):
        with open('../data/top10querylog.txt', 'a') as f:
            f.write('TOP10 QUERY IS BEING IN PROCESS !!!\n\n')
        '''
        self.embs = []
        self.labels = []
        cursors = self.mongo.mongodb_dashinfo.find({})
        unique_labels = [cursor['represent_image_id'] for cursor in cursors]
        cursors = self.mongo.mongodb_faceinfo.find({'image_id': {'$in': unique_labels}})
        for cursor in cursors:
            self.embs.append(np.array(cursor['embedding']))
            self.labels.append(cursor['image_id'])
        self.nof_registered_image_ids = len(self.labels)
        '''
        self.labels, self.embs = self.database.get_labels_and_embs_dashboard()
        self.matcher = FaissMatcher()
        self.matcher.fit(self.embs, self.labels)

        with open('../data/top10querylog.txt', 'a') as f:
            f.write('MATCHER BUILT!!!\n\n')

    def run(self):
        while not self.event.is_set():
            # first update check for new faces in dashboard
            '''
            if self.nof_registered_image_ids < self.mongo.mongodb_dashinfo.find({}).count():
                
                self.nof_registered_image_ids = self.mongo.mongodb_dashinfo.find({}).count()
                print('[Query TOP10] Update new registered image_id ...')
                cursors = self.mongo.mongodb_dashinfo.find({'represent_image_id': {'$nin': self.labels}})
                unique_labels = [cursor['represent_image_id'] for cursor in cursors]
                cursors = self.mongo.mongodb_faceinfo.find({'image_id': {'$in': unique_labels}})
                adding_embs = []
                adding_labels = []
                for cursor in cursors:
                    adding_embs.append(np.array(cursor['embedding']))
                    adding_labels.append(cursor['image_id'])
            '''
            adding_labels, adding_embs = self.database.get_labels_and_embs_dashboard(
                self.labels)
            if len(adding_labels) > 0:
                old_embs = self.embs
                old_labels = self.labels
                self.embs = old_embs + adding_embs
                self.labels = old_labels + adding_labels
                print('Find similar', len(adding_labels))
                self.matcher.update(adding_embs, adding_labels)

            # get new query from from queue, why not just trigger
            action_msg = self.rabbit_mq.receive_str(Config.Queues.ACTION)
            if action_msg is not None:
                return_dict = json.loads(action_msg)
                print('Receive: {}'.format(return_dict))
                if return_dict['actionType'] == 'getNearest':
                    data = return_dict['data']

                    results = {}
                    session_id = data['sessionId']
                    image_id = data['imageId']
                    print('[Query TOP10] image_id: ' + image_id)
                    with open('../data/top10querylog.txt', 'a') as f:
                        f.write('image_id: ' + image_id + '\n')

                    cursors = self.database.mongodb_faceinfo.find({'image_id': image_id})
                    if cursors.count() == 0:
                        print('[Query TOP10] THIS QUERY IMAGE ID HAS YET TO REGISTER')
                        with open('../data/top10querylog.txt', 'a') as f:
                            f.write('THIS QUERY IMAGE ID HAS YET TO REGISTER\n')
                        face_id = self.database.mongodb_dashinfo.find({'represent_image_id': image_id})[0]['face_id']
                        unique_labels = [
                            cursor['represent_image_id']
                            for cursor in self.database.mongodb_dashinfo.find({
                                'face_id':
                                face_id
                            })
                        ]
                        for label in unique_labels:
                            results[label] = '0'
                    else:
                        query_emb = cursors[0]['embedding']
                        embs = np.array(query_emb).astype('float32').reshape(
                            (-1, 128))
                        dists, inds = self.matcher._classifier.search(
                            embs, k=15)
                        dists = np.squeeze(dists)
                        inds = np.squeeze(inds)
                        top_match_ids = [self.labels[idx] for idx in inds]
                        for i, top_match_id in enumerate(top_match_ids):
                            if i < 11 and top_match_id != image_id:
                                results[top_match_id] = str(dists[i])
                    msg_results = {
                        'actionType': 'getNearest',
                        'sessionId': session_id,
                        'data': {
                            'images': results
                        }
                    }
                    with open('../data/top10querylog.txt', 'a') as f:
                        f.write('Result: \n{}\n\n'.format(results))
                    print('[Query TOP10] Result: \n{}'.format(results))
                    self.rabbit_mq.send_with_exchange(
                        Config.Queues.ACTION_RESULT, session_id,
                        json.dumps(msg_results))
            else:
                time.sleep(1)


class SplitMergeThread(Thread):

    def __init__(self, **args):
        self.database = args.get('database')
        self.rabbit_mq = args.get('rabbit_mq')
        self.matcher = args.get('matcher')
        self.event = threading.Event()
        super(SplitMergeThread, self).__init__()

    def run(self):
        while not self.event.is_set():
            # Get message from rabbit mq
            # TODO: Push new message instead of keeping pull when refactor RabbitMQ
            ms_msg = self.rabbit_mq.receive_str(Config.Queues.MERGE)
            ms_flag = 'merge'
            if ms_msg is None:
                ms_msg = self.rabbit_mq.receive_str(Config.Queues.SPLIT)
                ms_flag = 'split'

            if ms_msg is None:
                # skip if have no msg
                time.sleep(1)
                continue

            merge_anchor, merge_list = self.extract_info_from_json(ms_msg)
            print('Merge list', merge_list)
            while merge_list:
                image_id1 = merge_list.pop()
                self.merge_split(ms_flag, image_id1, merge_anchor)

    def join(self, timeout=None):
        self.event.set()
        super(SplitMergeThread, self).join(timeout)

    def merge_split(self, action_type, image_id1, face_id2):
        track_id1 = int(image_id1.split('_')[0])
        #track_id1_dir = os.path.join(Config.TRACKING_DIR, str(track_id1))
        '''
        cursors = self.mongo.mongodb_faceinfo.find({'track_id': track_id1})
        image_ids1 = [cursor['image_id'] for cursor in cursors] if cursors.count() != 0 else []
        cursors.rewind()
        is_registered = cursors[0]['is_registered']
        '''
        image_ids, labels, embs, is_registered = self.database.find_face_by_track_id(
            track_id1)

        print(image_id1, '|' * 50)
        '''
        old_face_id = self.mongo.mongodb_dashinfo.find({'represent_image_id': image_id1})[0]['face_id']
        '''
        old_face_id = self.database.find_face_id_by_image_id(image_id1)
        '''
        self.mongo.mongodb_mslog.insert_one(
            {
                'time_stamp': time.time(),
                'action_type': action_type,
                'image_id': image_id1,
                'old_face_id': old_face_id,
                'new_face_id': face_id2
            }
        )
        '''
        self.database.merge_split_log(action_type, image_id1, old_face_id,
                                      face_id2)
        '''
        self.mongo.mongodb_dashinfo.update({'track_id': track_id1}, {'$set':{'face_id': face_id2, 'is_registered': True}}, multi=True)
        '''
        self.database.update_face_id(image_ids, track_id1, face_id2)

        # TODO: update history in case the pickles have not written yet.
        #h_flag = tracker_manager.modify_id_step_by_step(track_id1, face_id2)
        #if h_flag == True:
        #    return True

        if not is_registered:
            # Update Matcher
            self.matcher.update(embs, labels)
        '''
        for label in image_ids1:
            self.mongo.mongodb_faceinfo.update({'image_id': label}, {'$set':{'face_id': face_id2, 'is_registered': True}}, multi=True)
        '''

        print('Done merge split', '=' * 20)

    # get info from file
    def extract_info_from_json(self, rbmq_msg):
        return_dict = json.loads(rbmq_msg)
        return_anchor = return_dict['visitorId']
        return_list = return_dict['images']
        return_list = [
            return_dir.split('/')[-1].replace('.jpg', '')
            for return_dir in return_list
        ]
        return return_anchor, return_list
