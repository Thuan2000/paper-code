import os
import time
from pipe import worker
from pipe.trace_back import process_traceback
from utils.logger import logger
from config import Config


class DashboardDatabaseWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')

    @process_traceback
    def doFaceTask(self, _task):
        start = time.time()
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        # ADAPT THIS FOR OTHERS SERVER
        # deleted_trackers = data['trackers']
        # for tracker in deleted_trackers
        tracker = data['tracker']
        if tracker.face_id == Config.Track.RECOGNIZED_FACE_ID:
            tracker.face_id = self.database.find_face_id_by_represent_image_id_in_faceinfo(
                tracker.represent_image_id)
            if tracker.face_id == None:
                raise Exception('represent_image_id in tracker not found!')

        for element in tracker.elements:
            self.database.insert_new_face(
                face_id=tracker.face_id,
                track_id=tracker.track_id,
                image_id=element.image_id,
                time_stamp=element.time_stamp,
                bounding_box=element.bounding_box.tolist(),
                embedding=element.embedding.tolist(),
                points=element.landmarks.tolist(),
                is_registered=tracker.is_registered,
                represent_image_id=tracker.represent_image_id)
            print('Save to database, face_id %s, track_id %s, image_id %s' \
                    % (tracker.face_id, tracker.track_id, element.image_id))
            logger.info('Save to database, face_id %s, track_id %s, image_id %s' \
                    % (tracker.face_id, tracker.track_id, element.image_id))


class AnnotationDatabaseWorker(worker.Worker):

    def __init__(self, **args):
        self.dataset_id = args.get('dataset_id')
        self.database = args.get('database')
        self.total_insert = 0

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        tracker = data['tracker']
        if tracker.face_id == Config.Track.INIT_FACE_ID or \
                tracker.face_id == Config.Track.BAD_TRACK:
            #tracker.face_id = tracker.generate_face_id('Anno', prefix='TCH')
            tracker.assign_face_id('Anno')

        for element in tracker.elements:

            self.database.insert_new_face(
                face_id=tracker.face_id,
                track_id=tracker.track_id,
                image_id=element.image_id,
                time_stamp=element.time_stamp,
                bounding_box=element.bounding_box.tolist(),
                embedding=element.embedding.tolist(),
                points=element.landmarks.tolist(),
                is_registered=tracker.is_registered,
                frame_id=element.frame_id)
            print('=' * 10, 'Save to db', element.image_id)
            logger.info('Save to database, face_id %s, track_id %s, image_id %s' \
                        % (tracker.face_id, tracker.track_id, element.image_id))


class RetentionDashboardDatabaseWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')
        self.socket = args.get('socket')

    @process_traceback
    def doFaceTask(self, _task):
        start = time.time()
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        # ADAPT THIS FOR OTHERS SERVER
        # deleted_trackers = data['trackers']
        # for tracker in deleted_trackers
        send_new_face = True
        tracker = data['tracker']
        if tracker.face_id == Config.Track.RECOGNIZED_FACE_ID:
            send_new_face = False
            tracker.face_id = self.database.find_face_id_by_represent_image_id_in_faceinfo(
                tracker.represent_image_id)
            if tracker.face_id == None:
                raise Exception('represent_image_id in tracker not found!')

        print("== %s: Update tracker %s to database" % (self.name, tracker.track_id))
        for element in tracker.elements:
            image_path = "/{}/{}/{}.jpg".format(Config.Source.AREA, tracker.track_id, element.image_id)
            paddedBoundingBox = [int(i) for i in element.str_padded_bbox.split("_")]
            self.database.insert_new_face(
                image=image_path,
                exportImage='',
                imageId=element.image_id,
                faceId=tracker.face_id,
                frameId=element.frame_id,
                boundingBox=element.bounding_box.tolist(),
                paddedBoundingBox=paddedBoundingBox,
                timestamp=element.time_stamp,
                embedding=element.embedding.tolist(),
                trackId=tracker.track_id,
                points=element.landmarks.tolist(),
                represent_image_id=tracker.represent_image_id,
                is_registered=tracker.is_registered)
            logger.info('Save to database, face_id %s, track_id %s, image_id %s' \
                    % (tracker.face_id, tracker.track_id, element.image_id))

        if hasattr(tracker, 'gender') and hasattr(tracker, 'age'):
            self.database.update_age_and_gender(tracker.face_id, tracker.gender, tracker.age)

        if send_new_face:
            self.socket.send_result(image_id=tracker.elements[0].image_id)
            print("== %s: Send face %s to dashboard" % (self.name, tracker.face_id))


class RetentionDashboardUpdateIsIgnore(worker.Worker):

    def __init__(self, **kwargs):
        self.database = kwargs.get('database')
        self.socket = kwargs.get('socket')

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        # ADAPT THIS FOR OTHERS SERVER
        # deleted_trackers = data['trackers']
        # for tracker in deleted_trackers
        tracker = data['tracker']
        print("== %s: Update tracker %s to database, isIgnored=%s" % (self.name, tracker.track_id, tracker.is_ignored))
        for element in tracker.elements:
            image_path = "/{}/{}/{}.jpg".format(Config.Source.AREA, tracker.track_id, element.image_id)
            paddedBoundingBox = [int(i) for i in element.str_padded_bbox.split("_")]
            self.database.insert_new_face(
                image=image_path,
                exportImage='',
                imageId=element.image_id,
                faceId=tracker.face_id,
                frameId=element.frame_id,
                boundingBox=element.bounding_box.tolist(),
                paddedBoundingBox=paddedBoundingBox,
                timestamp=element.time_stamp,
                embedding=element.embedding.tolist(),
                trackId=tracker.track_id,
                points=element.landmarks.tolist(),
                represent_image_id=tracker.represent_image_id,
                is_registered=tracker.is_registered)
            logger.info('Save to database, face_id %s, track_id %s, image_id %s' \
                    % (tracker.face_id, tracker.track_id, element.image_id))

        # update "isIgnored" in mongodb
        self.database.mongodb_info.update({"faceId": tracker.face_id},
                                          {"$set": {"isIgnored": tracker.is_ignored}},
                                          upsert=True)

        self.socket.send_result(image_id=tracker.elements[0].image_id)
        print("== %s: Send face %s to dashboard" % (self.name, tracker.face_id))
