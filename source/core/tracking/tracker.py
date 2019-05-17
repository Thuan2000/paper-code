from copy import deepcopy
from abc import ABCMeta, abstractmethod
import numpy as np
import os
import cv2
import time
import threading
import random
from scipy import misc
from core.cv_utils import create_if_not_exist, CropperUtils, find_most_common
from config import Config
from core.tracking.direction import calc_vectors_intersection

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class BaseTracker(metaclass=ABCMeta):

    def __init__(self, mean, covariance,
                 track_id, n_init=3,
                 max_age=30):
        self.track_id = track_id
        self.track_id_path = os.path.join(Config.Dir.TRACKING_DIR, str(self.track_id))
        # below is attribute for kalman filer tracker
        self._n_init = n_init
        self._max_age = max_age
        self.state = TrackState.Tentative

        self.mean = mean
        self.covariance = covariance
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        create_if_not_exist(self.track_id_path)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def random_samples(self, elements, rand_size=Config.Track.MAX_NROF_ELEMENTS):
        rand_size = min(len(elements), rand_size)
        return random.sample(elements, rand_size)

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, face_info):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, face_info.to_xyah())

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def get_pos(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    # only return good embs for tracker embs matching
    def get_embeddings(self):
        return np.array([])



class FaceTracker(BaseTracker):


    class TrackerInformation(object):
        """
        This class this create to clone necessary infomation for later step(matching, database),
        since don't need redundant attribute like Kalman filter mean and variances ...
        """

        def __init__(self, track_id, elements,
                    face_id, represent_image_id,
                    is_registered, track_id_path):
            self.track_id = track_id
            self.elements = elements
            self.face_id = face_id
            self.represent_image_id = represent_image_id
            self.is_registered = is_registered
            self.track_id_path = track_id_path

        def generate_face_id(self, prefix):
            return '{}-{}-{}'.format(prefix, self.track_id, time.time())


    def __init__(self, mean, covariance,
                 track_id, n_init=3,
                 max_age=30, _use_direction=Config.Track.USE_DIRECTION):
        self.track_id = track_id
        self.elements = []
        self.face_id = Config.Track.INIT_FACE_ID
        self.represent_image_id = None
        self.is_registered = True
        self.track_id_path = os.path.join(Config.Dir.TRACKING_DIR, str(self.track_id))
        self.use_direction = _use_direction

        # Here is the new attribute added this time
        # Since now we keep bad detections low coeff to track by iou
        # so we need to keep track of good faces to send
        self.nrof_good_faces = 0
        # This attribute to serve RealTime tracker - we send tracker to matching two times: t1 and t2
        # t1 is send when we we have n if good face, so t2 don't need to resend what t1 is already do
        self.sent_elements_index = 0

        # below is attribute for kalman filer tracker
        self.state = TrackState.Tentative

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.directions = []

        super(FaceTracker, self).__init__(mean, covariance, track_id, n_init=3, max_age=30)
        create_if_not_exist(self.track_id_path)

    def update_embeddings(self, embeddings_array, start_id, interval):
        j = 0
        for i in range(start_id, start_id + interval):
            self.elements[i].update_embedding(embeddings_array[j])
            j += 1

    def update(self, kf, face_info):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, face_info.to_xyah())

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # TODO: refactor this to storage step, but still have to guaranty the same speed

        image_id = '{}_{}'.format(self.track_id, face_info.str_info())
        image_name = image_id + '.jpg'
        img_path = os.path.join(self.track_id_path, image_name)
        save_image = cv2.cvtColor(face_info.face_image, cv2.COLOR_BGR2RGB)
        t = threading.Thread(
            target=(cv2.imwrite), args=(
                img_path,
                save_image))
        t.start()

        face_info.image_id = image_id
        self.elements.append(face_info)
        if face_info.is_good():
            self.nrof_good_faces +=1
        if self.use_direction and len(self.elements) > 2:
            direction = calc_vectors_intersection((self.elements[-2].centroid, self.elements[-1].centroid), \
                                        Config.ROI.ROI_CORDINATE[Config.ROI.USE], \
                                        Config.ROI.LINE_CORDINATE[Config.ROI.USE])
            if direction is not None and direction[0]:
                self.directions.append(direction[1])



    def is_qualified_to_be_recognized(self):

        if self.use_direction and len(self.directions) > 0:
            match_direction = find_most_common(self.directions) == Config.Track.DIRECTION
        else:
            match_direction = True

        return self.nrof_good_faces >= Config.Track.MIN_NOF_TRACKED_FACES and match_direction and \
            self.face_id == Config.Track.INIT_FACE_ID

    def is_recognized(self):
        # Only tracker that batch 2 have good faces will be qualified,
        # we only check if it exist a good face, that's enough
        if self.face_id == Config.Track.RECOGNIZED_FACE_ID:
            for element in self.elements[self.sent_elements_index:]:
                if element.is_good():
                    return True
        # if we use full-track tracking, this will always return True
        return False

    def clear_elements(self):
        self.elements.clear()

    def get_embeddings(self):
        # TODO: decide to only return good faces or not
        # all face: fast return, maybe bad matching result
        # good face: slow return, may be good matching result
        # embs = [e.embedding for e in self.elements if e.is_good()]
        embs = [e.embedding for e in self.elements if e.embedding is not None]
        return embs

    def clone(self):
        # we got two mode here, early recognition and deleted
        if self.face_id == Config.Track.INIT_FACE_ID:
            elements = [e for e in self.elements if e.is_good()]
        elif self.face_id == Config.Track.RECOGNIZED_FACE_ID:
            elements = [e for e in self.elements[self.sent_elements_index:] if e.is_good()]
        elements = self.random_samples(elements)

        _tracker = FaceTracker.TrackerInformation(
            track_id=self.track_id,
            elements=elements,
            face_id=self.face_id,
            represent_image_id= self.represent_image_id,
            is_registered=self.is_registered,
            track_id_path=self.track_id_path)

        return _tracker

