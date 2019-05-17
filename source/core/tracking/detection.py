import time
from copy import deepcopy
import numpy as np
from scipy import misc
from config import Config
from core.cv_utils import create_if_not_exist, padded_landmark


class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


# TODO: Remove FaceInfo, too ambiguos, move to its own class
class FaceInfo:
    '''
    Contain information about face in frame, identify using frame id
    And bounding box, landmark for cropping face from frame,
    Lazily crop frame when needed
    '''

    def __init__(
            self,
            _bounding_box,
            _bbox_confident,
            _frame_id,
            _face_image,
            _str_padded_bbox,
            _landmarks):
        '''
        Wrapping for face
        :param frame_id: check which frame this frame is comming from
        :param bouding_box: (top, left, bottom, right) -> face location in frame
        :param landmark: landmarks in face: 2 eyes, nose, 2 side mouth
        '''
        # assign value
        self.bounding_box = _bounding_box
        self.bbox_confident = _bbox_confident
        self.embedding = None
        self.frame_id = _frame_id
        self.face_image = _face_image
        self.str_padded_bbox = _str_padded_bbox
        self.time_stamp = _frame_id
        self.landmarks = padded_landmark(_landmarks, self.bounding_box, self.str_padded_bbox, Config.Align.MARGIN)
        self.image_id = None
        self.quality = 100
        self.yaw_angle = self.calc_face_angle(self.landmarks)
        self.pitch_angle = self.calc_face_pitch(self.landmarks)
        self.tlwh = np.asarray(self.to_tlwh(), dtype=np.float)
        self.centroid = np.asarray(self.to_centroid(), dtype=np.float)


    def update_embedding(self, _embedding):
        self.embedding = _embedding

    def set_face_quality(self, face_quality):
        self.quality = face_quality
        # check if this face is good

    def is_good(self):
        return abs(self.yaw_angle) < Config.Filters.YAW and \
                abs(self.pitch_angle) < Config.Filters.PITCH and \
                self.quality > Config.Filters.COEFF

    def str_info(self):
        bbox_str = '_'.join(
            np.array(self.bounding_box, dtype=np.unicode).tolist())
        str_info = '{}_{}_{}'.format(bbox_str, self.frame_id,
                                     self.str_padded_bbox)
        return str_info

    def angle_between(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def calc_face_angle(self, points):
        point_x = points[0:5]
        point_y = points[5:10]

        if point_x[0] > point_x[2]:
            left_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0)))
        else:
            left_angle = np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0)))
        if point_x[1] < point_x[2]:
            right_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        else:
            right_angle = np.degrees(
                self.angle_between(
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        tmp_beta = left_angle - right_angle

        return tmp_beta

    def calc_face_pitch(self, points):
        point_x = points[0:5]
        point_y = points[5:10]

        if point_x[0] > point_x[2]:
            top_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0)))
        else:
            top_angle = np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0)))
        if point_x[1] < point_x[2]:
            bottom_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        else:
            bottom_angle = np.degrees(
                self.angle_between(
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        tmp_beta = top_angle - bottom_angle

        return tmp_beta

    def deep_clone(self):
        return deepcopy(self)


    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_tlwh(self):
        '''
        definition:
                print(rect) --> (x,y,w,h)
                print(self.bounding_box)   --> (x,y,x1,y1)
        '''
        ret = self.bounding_box.copy()
        ret[2] = ret[2] - ret[0]
        ret[3] = ret[3] - ret[1]
        return ret

    def to_centroid(self):
        ret = self.bounding_box.copy()
        ret = [ret[0] + (ret[2] / 2), ret[1] + (ret[3] / 2)]
        return ret

class PedestrianInfo:
    '''
    Contain information about face in frame, identify using frame id
    And bounding box, landmark for cropping face from frame,
    Lazily crop frame when needed
    '''

    def __init__(
            self,
            _bounding_box,
            _time,
            _face_image,
            _str_padded_bbox,
            quality):
        '''
        Wrapping for face
        :param frame_id: check which frame this frame is comming from
        :param bouding_box: (top, left, bottom, right) -> face location in frame
        :param landmark: landmarks in face: 2 eyes, nose, 2 side mouth
        '''
        # assign value
        self.bounding_box = _bounding_box
        self.embedding = None
        # Use time as frame ID
        self.frame_id = _time
        self.face_image = _face_image
        self.str_padded_bbox = _str_padded_bbox
        self.time_stamp = _time
        self.image_id = None
        self.quality = 100
        self.landmarks = np.array([])
        self.tlwh = np.asarray(self.to_tlwh(), dtype=np.float)
        self.centroid = np.asarray(self.to_centroid(), dtype=np.float)


    def update_embedding(self, _embedding):
        self.embedding = _embedding

        # check if this face is good

    def is_good(self):
        return self.quality > Config.Filters.COEFF

    def str_info(self):
        bbox_str = '_'.join(
            np.array(self.bounding_box, dtype=np.unicode).tolist())
        str_info = '{}_{}_{}'.format(bbox_str, self.frame_id,
                                     self.str_padded_bbox)
        return str_info

    def angle_between(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def calc_face_angle(self, points):
        point_x = points[0:5]
        point_y = points[5:10]

        if point_x[0] > point_x[2]:
            left_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0)))
        else:
            left_angle = np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0)))
        if point_x[1] < point_x[2]:
            right_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        else:
            right_angle = np.degrees(
                self.angle_between(
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        tmp_beta = left_angle - right_angle

        return tmp_beta

    def calc_face_pitch(self, points):
        point_x = points[0:5]
        point_y = points[5:10]

        if point_x[0] > point_x[2]:
            top_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0)))
        else:
            top_angle = np.degrees(
                self.angle_between(
                    (point_x[0] - point_x[2], point_y[0] - point_y[2], 0),
                    (point_x[1] - point_x[2], point_y[1] - point_y[2], 0)))
        if point_x[1] < point_x[2]:
            bottom_angle = 360 - np.degrees(
                self.angle_between(
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        else:
            bottom_angle = np.degrees(
                self.angle_between(
                    (point_x[3] - point_x[2], point_y[3] - point_y[2], 0),
                    (point_x[4] - point_x[2], point_y[4] - point_y[2], 0)))
        tmp_beta = top_angle - bottom_angle

        return tmp_beta

    def deep_clone(self):
        return deepcopy(self)


    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_tlwh(self):
        '''
        definition:
                print(rect) --> (x,y,w,h)
                print(self.bounding_box)   --> (x,y,x1,y1)
        '''
        ret = self.bounding_box.copy()
        ret[2] = ret[2] - ret[0]
        ret[3] = ret[3] - ret[1]
        return ret

    def to_centroid(self):
        ret = self.bounding_box.copy()
        ret = [ret[0] + (ret[2] / 2), ret[1] + (ret[3] / 2)]
        return ret