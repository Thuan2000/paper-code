import time
import numpy as np
from collections import Counter
from pipe import worker, task
from pipe.trace_back import process_traceback
from core import major_distance
from core.cv_utils import get_avg_dists
from utils.logger import logger
from config import Config


class MatchingWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')
        self.global_matcher = args.get('matcher')
        self.area = args.get('area')
        super(MatchingWorker, self).__init__()

    @process_traceback
    def doFaceTask(self, _task):
        start = time.time()
        data = _task.depackage()
        tracker = data['tracker']
        self.handleNewTracker(tracker)

    def handleNewTracker(self, tracker):
        if tracker.face_id == Config.Track.INIT_FACE_ID:
            predicted_face_id, _ = self.global_recognize_tracker(tracker)
            if predicted_face_id == Config.Matcher.NEW_FACE:
                predicted_face_id = tracker.generate_face_id(prefix=self.area)
                logger.info("Generated face id: %s" % (predicted_face_id))
                tracker.is_registered = False
                embs = []
                labels = []
                for element in tracker.elements:
                    embs.append(element.embedding)
                    labels.append(element.image_id)
                self.global_matcher.update(embs, labels)
            tracker.face_id = predicted_face_id
            logger.info("Recognized a new tracker face id: %s" % (tracker.face_id))
            print("== %s: Recognized new tracker as: %s" % (self.name, tracker.face_id))
        if tracker.face_id == Config.Track.RECOGNIZED_FACE_ID:
            logger.info("Handle remaining elements of sent tracker")
            print('== %s: Passing tracker %s to query face_id' % (self.name, tracker.track_id))

        _task = task.Task(task.Task.Face)
        _task.package(tracker=tracker,
                     type=Config.Worker.TASK_TRACKER)
        self.putResult(_task)

    # TODO: this functions below might deserved it's own class
    def global_recognize_tracker(self, tracker):
        predicted_ids = []
        predicted_dists = []
        avg_dists = {}
        # change default from init to new_face
        # top_predict_id = Config.Track.INIT_FACE_ID
        top_predict_id = Config.Matcher.NEW_FACE
        for element in tracker.elements:
            top_ids, dists = self.global_matcher.match(
                element.embedding, return_dists=True)
            if len(dists) == 0 or dists[0] == -1 \
                    or dists[0] > Config.Matcher.MIN_ASSIGN_THRESHOLD:
                predicted_ids.append(Config.Matcher.NEW_FACE)
                predicted_dists.append(-1)
            else:
                predicted_ids.append(top_ids[0])
                predicted_dists.append(dists[0])

        predicted_face_ids = self._query_face_ids(predicted_ids)

        if len(predicted_face_ids) > 0:
            most_popular_id, nof_predicted = Counter(
                predicted_face_ids).most_common(1)[0]
            # Count number of valid elements
            nrof_valid_elements = 0
            for i, _id in enumerate(predicted_face_ids):
                if _id == most_popular_id and predicted_dists[
                        i] < Config.Matcher.MIN_ASSIGN_THRESHOLD:
                    nrof_valid_elements += 1

            # Check if the predicted ID satisfies Major rate and valid rate
            if nof_predicted / len(predicted_face_ids) >= Config.Track.HISTORY_RETRACK_MINRATE \
                    or nrof_valid_elements / nof_predicted >= Config.Track.VALID_ELEMENT_MINRATE:
                top_predict_id = most_popular_id
            avg_dists = get_avg_dists(predicted_face_ids, predicted_dists)

        return top_predict_id, avg_dists

    def _query_face_ids(self, predicted_ids):
        predicted_face_ids = []
        for predicted_id in predicted_ids:
            if predicted_id != Config.Matcher.NEW_FACE:
                predicted_face_id = self.database.find_face_id_by_image_id_in_faceinfo(
                    predicted_id)
                if predicted_face_id is not None:
                    predicted_face_ids.append(predicted_face_id)
                else:
                    predicted_face_ids.append(Config.Matcher.NEW_FACE)
            else:
                predicted_face_ids.append(Config.Matcher.NEW_FACE)
        return predicted_face_ids


class MasanMatchingWorker(worker.Worker):

    def __init__(self, **args):
        self.database = args.get('database')
        self.socket = args.get('socket')
        self.area = args.get('area')
        self.use_blacklist = args.get('use_blacklist', False)
        self.blacklist = ([], np.array([]))
        if self.use_blacklist:
            self.get_blacklist()
        super(MasanMatchingWorker, self).__init__()

    def get_blacklist(self, key='isIgnored'):
        labels, embs = self.database.get_labels_and_embs_by_value_is_true_in_info(key)
        embs = embs.reshape(-1, 128)
        self.blacklist = (labels, embs)
        print('Updated blacklist')

    @process_traceback
    def doFaceTask(self, task):
        if self.use_blacklist:
            if self.socket.get_update_blacklist():
                self.get_blacklist()
        start = time.time()
        data = task.depackage()
        tracker = data['tracker']
        self.handleNewTracker(tracker)

    def handleNewTracker(self, tracker):
        if tracker.face_id == Config.Track.INIT_FACE_ID:
            predicted_face_id, _ = self.blacklist_recognize_tracker(tracker)
            if predicted_face_id == Config.Matcher.NEW_FACE:
                predicted_face_id = tracker.generate_face_id(prefix=self.area)
                logger.info("Generated face id: %s" % (predicted_face_id))
                tracker.is_registered = False
            tracker.face_id = predicted_face_id
            logger.info("Recognized a new tracker face id: %s" % (tracker.face_id))
            print("== %s: Recognized new tracker as: %s" % (self.name, tracker.face_id))
        if tracker.face_id == Config.Track.RECOGNIZED_FACE_ID:
            logger.info("Handle remaining elements of sent tracker")
            print('== %s: Passing tracker %s to query face_id' % (self.name, tracker.track_id))
        embs = []
        labels = []
        for element in tracker.elements:
            embs.append(element.embedding)
            labels.append(element.image_id)

        _task = task.Task(task.Task.Face)
        _task.package(tracker=tracker,
                     type=Config.Worker.TASK_TRACKER)
        self.putResult(_task)

    # TODO: this functions below might deserved it's own class
    def blacklist_recognize_tracker(self, tracker):
        predicted_ids = []
        predicted_dists = []
        avg_dists = {}
        elements_ems = []
        # change default from init to new_face
        # top_predict_id = Config.Track.INIT_FACE_ID
        top_predict_id = Config.Matcher.NEW_FACE
        for element in tracker.elements:
            elements_ems.append(element.embedding)
        elements_ems = np.array(elements_ems)
        print(elements_ems.shape)
        print(self.blacklist[1].shape)
        predicted_ids, predicted_dists = major_distance.match(
            self.blacklist[1], elements_ems, self.blacklist[0])
        for i, dist in enumerate(predicted_dists):
            if dist > Config.Matcher.MIN_ASSIGN_THRESHOLD:
                predicted_ids[i] = Config.Matcher.NEW_FACE

        predicted_face_ids = self._query_face_ids(predicted_ids)

        if len(predicted_face_ids) > 0:
            most_popular_id, nof_predicted = Counter(
                predicted_face_ids).most_common(1)[0]
            # Count number of valid elements
            nrof_valid_elements = 0
            for i, _id in enumerate(predicted_face_ids):
                if _id == most_popular_id and predicted_dists[
                        i] < Config.Matcher.MIN_ASSIGN_THRESHOLD:
                    nrof_valid_elements += 1

            # Check if the predicted ID satisfies Major rate and valid rate
            if nof_predicted / len(predicted_face_ids) >= Config.Track.HISTORY_RETRACK_MINRATE \
                    or nrof_valid_elements / nof_predicted >= Config.Track.VALID_ELEMENT_MINRATE:
                top_predict_id = most_popular_id
            avg_dists = get_avg_dists(predicted_face_ids, predicted_dists)

        return top_predict_id, avg_dists

    def _query_face_ids(self, predicted_ids):
        predicted_face_ids = []
        for predicted_id in predicted_ids:
            if predicted_id != Config.Matcher.NEW_FACE:
                predicted_face_id = self.database.find_face_id_by_image_id_in_faceinfo(
                    predicted_id)
                if predicted_face_id is not None:
                    predicted_face_ids.append(predicted_face_id)
                else:
                    predicted_face_ids.append(Config.Matcher.NEW_FACE)
            else:
                predicted_face_ids.append(Config.Matcher.NEW_FACE)
        return predicted_face_ids
