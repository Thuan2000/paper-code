from core.tracking.kalman import KalmanBoxTracker

import cv2
import dlib
from abc import ABCMeta, abstractmethod
from config import Config


def bounding_box_to_rect(bb):
    '''
	   definition:
	   		print(rect) --> (x,y,w,h)
	   		print(bb)   --> (x,y,x1,y1)
	'''
    return (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])


def rect_to_bounding_box(rect):
    '''
	   definition:
	   		print(rect) --> (x,y,w,h)
	   		print(bb)   --> (x,y,x1,y1)
	'''
    return (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])


class AbstractTracker(metaclass=ABCMeta):

    def __init__(self):
        self.coords = None
        self.dormant = False
        pass

    @abstractmethod
    def start_track(self, frame, bounding_box):
        pass

    def predict(self):
        pass

    def padding(self, bb, pad_horizontal, pad_vertical):
        return (int(max(0, bb[0] - pad_horizontal)), int(max(0, bb[1])),
                int(bb[2] + pad_horizontal), int(bb[3] + pad_vertical))

    def get_bb(self):
        return self.coords

    def replace_track(self, frame, bounding_box):
        self.start_track(frame, bounding_box)


'''
	These following classes are wrappers for various different trackers
	**Help generalizing them*
'''


class KCFTracker(AbstractTracker):

    def __init__(self, pad_vertical=0, pad_horizontal=0):
        '''
			@param:
				pad_vertical = vertical pad
				pad_horizontal = horizontal pad
			definition:
				Padding makes it easier to track small target.
		'''
        self.tracker = None
        self.coords = None
        self.pad_vertical = pad_vertical
        self.pad_horizontal = pad_horizontal
        self.KalmanBoxTracker = None

    def start_track(self, frame, bb):

        self.tracker = cv2.TrackerKCF_create()
        self.dormant = False
        #Padding makes it easier to track small target.
        bb = self.padding(bb, self.pad_horizontal, self.pad_vertical)
        #opencv 2drect is in the form of (x,y,w,h)
        rect = bounding_box_to_rect(bb)
        self.coords = bb
        _ = self.tracker.init(frame, rect)

    def predict(self, frame):
        '''
			@param:
				frame: the current image of the frame

			@return:
				False -- Tracker has lost track of the target
				True -- otherwise
		'''
        assert self.tracker is not None

        _, rect = self.tracker.update(frame)

        self.dormant = not _

        if (_):  #return true if tracker has not lost track
            self.coords = rect_to_bounding_box(rect)
        return _

    def get_bb(self):
        '''
			@return: bounding box of the tracking target
			bb = (x1,y1,x2,y2)
		'''
        assert self.coords is not None
        #get original before-pad bounding box
        return self.padding(self.coords, -self.pad_horizontal,
                            -self.pad_vertical)


class KalmanTracker(AbstractTracker):

    def __init__(self):
        self.KalmanBoxTracker = None
        self.coords = None

    def start_track(self, frame, bb):
        self.dormant = False
        self.KalmanBoxTracker = KalmanBoxTracker((bb[0], bb[1], bb[2], bb[3]))
        self.coords = bb

    def predict(self, frame):
        '''
			@param:
				frame: the current image of the frame

			@return:
				Always return True.
				Kalman Tracker doesn't have false report
		'''
        bb = self.KalmanBoxTracker.predict()[0]
        self.coords = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        self.dormant = False
        return True

    def replace_track(self, frame, bb):
        self.KalmanBoxTracker.predict((bb[0], bb[1], bb[0] + bb[2],
                                       bb[1] + bb[3]))


class DlibTracker(AbstractTracker):

    def __init__(self, pad_vertical=0, pad_horizontal=0):
        self.tracker = None
        self.coords = None
        self.pad_vertical = pad_vertical
        self.pad_horizontal = pad_horizontal

    def start_track(self, frame, bb):
        self.tracker = dlib.correlation_tracker()
        self.dormant = False
        bb = self.padding(bb, self.pad_horizontal, self.pad_vertical)
        self.coords = bb
        self.tracker.start_track(frame,
                                 dlib.rectangle(bb[0], bb[1], bb[2], bb[3]))

    def predict(self, frame):
        assert self.tracker is not None
        _ = self.tracker.update(frame)  # _ == confident value

        rect = self.tracker.get_position()
        self.coords = (int(rect.left()), int(rect.top()), int(rect.right()),
                       int(rect.bottom()))

        self.dormant = _ <= Config.Track.DLIB_TRACK_QUALITY
        return _ > Config.Track.DLIB_TRACK_QUALITY

    def get_bb(self):
        assert self.coords is not None
        return self.padding(self.coords, -self.pad_horizontal,
                            -self.pad_vertical)
