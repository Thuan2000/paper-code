import os
import dlib
import time
import json
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import Counter
from scipy import misc
from core.cv_utils import check_overlap, get_img_url_by_id
from core.cv_utils import print_out_frequency, get_avg_dists, calc_iou
from core.tracking import kalman_filter
from core.tracking.linear_assignment import matching_cascade, min_cost_matching
from core.tracking.linear_assignment import gate_cost_matrix
from core.tracking import iou_matching
from core.tracking import tracker
from core.tracking import nn_matching
from config import Config


class BaseTrackerManager(metaclass=ABCMeta):

    def __init__(self,
                 metric=None,
                 current_id=0,
                 max_age=30, n_init=3,
                 max_iou_distance=Config.Track.MAX_IOU_DISTANCE):
        self.id_counter = current_id
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.trackers = []
        if metric == None:
            self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",
                Config.Track.MAX_COSINE_DISTANCE, Config.Track.NN_BUDGET)
        self.kf = kalman_filter.KalmanFilter()

    def predict(self):
        for tracker in self.trackers:
            tracker.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.

        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(
                self.kf, detections[detection_idx])
        for tracker_idx in unmatched_tracks:
            self.trackers[tracker_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_tracker(detections[detection_idx])

        # Update distance metric.
        # TODO: validate active_targets in unused in original code
        # active_targets = [tracker.track_id for tracker in self.trackers if tracker.is_confirmed()]
        features, targets = [], []
        for tracker in self.trackers:
            if not tracker.is_confirmed():
                continue
            feature = tracker.get_embeddings()
            if feature is not None:
                features += feature
                targets += [tracker.track_id for _ in feature]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), targets)

        # Only return trackers that is good enough for matching
        deleted_trackers = [t for t in self.trackers if t.is_deleted()]
        self.trackers = [t for t in self.trackers if not t.is_deleted()]
        return deleted_trackers

    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.trackers) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.trackers) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_trackers_a, unmatched_detections = \
            matching_cascade(
                self.embs_cost_matrix, self.metric.matching_threshold, self.max_age,
                self.trackers, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_trackers_a if
            self.trackers[k].time_since_update == 1]
        unmatched_trackers_a = [
            k for k in unmatched_trackers_a if
            self.trackers[k].time_since_update != 1]
        matches_b, unmatched_trackers_b, unmatched_detections = \
            min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.trackers,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_trackers = list(set(unmatched_trackers_a + unmatched_trackers_b))
        return matches, unmatched_trackers, unmatched_detections

    # Default method, if this is not implemented then tracking by emb will be skipped
    def embs_cost_matrix(self, trackers, detections, tracker_indices, detection_indices):
        return np.array([])

    @abstractmethod
    def _initiate_tracker(self, detection):
        pass


class FaceTrackerManager(BaseTrackerManager):

    def __init__(self,
                 metric,
                 current_id=0,
                 max_age=30, n_init=3,
                 max_iou_distance=Config.Track.MAX_IOU_DISTANCE):
        super(FaceTrackerManager, self).__init__(metric, current_id, max_age, n_init, max_iou_distance)

    def update(self, detections):
        deleted_trackers = super().update(detections)
        qualified_trackers, disqualified_trackers = self.process_deleted_trackers(deleted_trackers)
        return qualified_trackers, disqualified_trackers

    def process_deleted_trackers(self, deleted_trackers):
        qualified_trackers = []
        disqualified_trackers = []
        for tracker in deleted_trackers:
            if tracker.is_qualified_to_be_recognized() or tracker.is_recognized():
                qualified_trackers.append(tracker.clone())
            else:
                disqualified_trackers.append(tracker.clone())
        return qualified_trackers, disqualified_trackers

    # find trackers that's qualified for recognition
    def get_early_qualified_trackers(self):
        qualified_trackers = []
        for tracker in self.trackers:
            if tracker.is_qualified_to_be_recognized():
                # set index and face_id to cutoff tracker when we send at it's delete
                tracker.represent_image_id = tracker.elements[0].image_id
                qualified_trackers.append(tracker.clone())
                tracker.face_id = Config.Track.RECOGNIZED_FACE_ID
                tracker.sent_elements_index = len(tracker.elements)
        return qualified_trackers

    def embs_cost_matrix(self, trackers, detections, tracker_indices, detection_indices):
        embs = np.array([detections[i].embedding for i in detection_indices])
        targets = np.array([trackers[i].track_id for i in tracker_indices])
        cost_matrix = self.metric.distance(embs, targets)
        cost_matrix = gate_cost_matrix(
            self.kf, cost_matrix, trackers, detections, tracker_indices,
            detection_indices)
        return cost_matrix

    def _initiate_tracker(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.id_counter += 1
        self.trackers.append(tracker.FaceTracker(
            mean, covariance, self.id_counter, self.n_init, self.max_age))


class ObjectTrackerManager(BaseTrackerManager):

    def get_confirmed_trackers(self):
        trackers = [t for t in self.trackers if t.is_confirmed()]
        return trackers

    def _initiate_tracker(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.id_counter += 1
        self.trackers.append(tracker.BaseTracker(
            mean, covariance, self.id_counter, self.n_init, self.max_age))
