"""
This script contains all handlers for reading image frames from different sources
"""
from config import Config
import os
import glob
from abc import ABCMeta, abstractmethod
import cv2
from scipy import misc
# from queue import Queue
import queue
from multiprocessing import Process, current_process, Queue, log_to_stderr
import multiprocessing
import pdb
import time


class AbstractFrameReader(metaclass=ABCMeta):
    '''
    Abstract class to provide frame
    '''

    def __init__(self, scale_factor=1, should_crop=False):
        '''
        :param scale_factor: reduce frame factor
        '''
        self._scale_factor = scale_factor
        self._should_crop = should_crop

        # for running in new thread
        logger = log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)
        self._stopped = False
        self._queue = Queue(128)
        p = Process(target=self._read_frame, name='read_frame_process')
        p.start()
        #p.join()

    @abstractmethod
    def _frame(self):
        '''
        get actual frame, base on implementation
        '''
        pass

    def _read_frame(self):
        '''
        Continously read frame and add it to queue
        in background thread
        '''
        while True:
            if self._stopped:
                print('Stop reading new frame')
                return

            if not self._queue.full():
                frame = self._frame()
                if frame is None:
                    # None can be add to the queue
                    continue
                self._queue.put(frame)
            else:  # if queue is full, sleep for 5s
                time.sleep(4)
                print('Sleep')

    def next_frame(self):
        '''
        Return next frame, None if empty
        '''
        try:
            return self._queue.get(False)
        except queue.Empty:
            return None

    def has_next(self):
        '''
        Check if has more frame to read
        '''
        return True

    def _scale_frame(self, frame):
        '''
        Rescale frame for faster processing
        :param frame: input frame to rescale
        :return frame: resized frame
        '''
        if frame is None:
            return None

        if self._scale_factor > 1:
            frame = cv2.resize(frame, (int(len(frame[0]) / self._scale_factor),
                                       int(len(frame) / self._scale_factor)))
        return frame

    def _crop_roi(self, frame):
        '''
        Crop roi for faster downstream processing
        :param frame: input frame to crop
        :return frame: cropped frame
        '''
        if frame is None:
            return None

        if self._should_crop:
            roi = AbstractFrameReader._roi(frame, self._scale_factor)
            frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
        return frame

    @staticmethod
    def _roi(frame, scale):
        '''
        Find ROI of interest for particular frame
        '''
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        roi = [
            int(center[0] - Config.Frame.ROI_CROP[0] * w),
            int(center[1] - Config.Frame.ROI_CROP[1] * h),
            int(center[0] + Config.Frame.ROI_CROP[2] * w),
            int(center[1] + Config.Frame.ROI_CROP[3] * h)
        ]
        return roi

    # @abstractmethod
    def release(self):
        '''
        Release this frame reader
        '''
        pass

    def get_info(self):
        '''
        Get current frame reader info
        '''
        pass


class RabbitFrameReader(AbstractFrameReader):
    '''
    Read frame from rabbit queue, for register for live image
    '''

    def __init__(self,
                 rb,
                 queue_name,
                 scale_factor=Config.Frame.SCALE_FACCTOR,
                 should_crop=False):
        '''
        :param rb: rabbitmq instance to read new frame from
        :param scale_factor: reduce frame factor
        :param should_crop: should crop roi
        '''
        self.__rb = rb
        self.__queue_name = queue_name
        if self.__rb.is_exist(self.__queue_name):
            raise ValueError("Queue not exist")
        super(RabbitFrameReader, self).__init__(scale_factor, should_crop)

    def _frame(self):
        '''
        Read next frame from rabbit, may return None if there is no frame avaiable
        '''
        frame = self.__rb.receive_raw_live_image(self.__queue_name)
        frame = self._crop_roi(frame)
        frame = self._scale_frame(frame)
        return frame

    def release(self):
        self._stopped = True
        pass

    def get_info(self):
        '''
        Get information, omit one frame to get info
        '''
        frame = None
        while frame is None:
            frame = self.next_frame()
        return 24, frame.shape[1], frame.shape[0]


class URLFrameReader(AbstractFrameReader):
    """
    Read frame from video stream or from video path or web cam
    """

    # TODO: Should create 3 subclass?
    WEBCAM = 0
    VIDEO_FILE = 1
    IP_STREAM = 2

    def __init__(self,
                 cam_url,
                 scale_factor=Config.Frame.SCALE_FACCTOR,
                 should_crop=False):
        '''
        :param cam_url: url for video stream
        :param scale_factor: reduce frame factor
        '''
        if type(cam_url) == int or cam_url.isdigit():
            cam_url = int(cam_url)
            self.__url_type = URLFrameReader.WEBCAM
        elif self.__is_file(cam_url):
            self.__url_type = URLFrameReader.VIDEO_FILE
        else:
            self.__url_type = URLFrameReader.IP_STREAM
        self.__cam_url = cam_url

        # temporal video to get information
        video = cv2.VideoCapture(cam_url)
        self.__frame_index = video.get(cv2.CAP_PROP_POS_FRAMES)
        self.__fps = video.get(cv2.CAP_PROP_FPS)
        self.__total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)

        super(URLFrameReader, self).__init__(scale_factor, should_crop)

    def _read_frame(self):
        # create video reader in reading process
        self.__video_capture = cv2.VideoCapture(self.__cam_url)
        super(URLFrameReader, self)._read_frame()
        self.__video_capture.release()

    def _frame(self):
        '''
        Return scaled frame from video stream
        '''
        # read frame using opencv, may skip some frame if reading too fast
        curr_frame = self.__video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        succ, frame = self.__video_capture.read()
        if not succ:
            self.__video_capture.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self._crop_roi(frame)
        frame = self._scale_frame(frame)
        return frame

    def next_frame(self):
        frame = super(URLFrameReader, self).next_frame()
        if frame is not None:
            # update frame position
            self.__frame_index += 1

        return frame

    def has_next(self):
        '''
        If the video stream is limited (like video file),
            the frame will return False when it's read all the frame
        else (ip stream or webcam) it will always return True
        '''
        if self.__url_type == URLFrameReader.VIDEO_FILE:
            # print('Has more', self.__frame_index, self.__total_frame)
            return self.__frame_index \
                    < self.__total_frame
        else:
            return True

    def release(self):
        '''
        Release the video source
        '''
        self._stopped = True

    def __is_file(self, cam_url):
        '''
        Check if cam_url is a video in filesystem or an ip stream
        :param cam_url: url to check
        :return True if cam_url exist in file system
        '''
        return os.path.exists(cam_url)

    def get_info(self):
        '''
        Frame is now depend also on ROI, not just simple scale factor
        Get information, omit one frame to get info
        '''
        frame = None
        while frame is None:
            frame = self.next_frame()

        return self.__fps, \
            frame.shape[1], frame.shape[0]


class DirectoryFrameReader(AbstractFrameReader):
    """
    Read all image from a directory

    >>> frame_reader = DirectoryFrameReader(r'../data/matching/set1', 'jpg'); \
        len(frame_reader._DirectoryFrameReader__image_files)
    9
    >>> for i in range(9):
    ...     _ = frame_reader.next_frame()
    >>> frame_reader.has_next()
    False
    """

    def __init__(self,
                 dir_,
                 ext='jpg',
                 scale_factor=Config.Frame.SCALE_FACCTOR,
                 should_crop=False):
        '''
        :param dir_: directory that contains images
        :pram ext: image extension to read
        :param scale_factor: reduce frame factor
        '''
        self.__image_files = glob.glob(os.path.join(dir_, '*.%s' % ext))
        self.__image_files.sort()
        self.__frame_index = 0
        super(DirectoryFrameReader, self).__init__(scale_factor, should_crop)

    def _frame(self):
        '''
        Read next image from directory
        '''
        frame = misc.imread(self.__image_files[self.__frame_index])
        frame = self._scale_frame(frame)
        frame = self._crop_roi(frame)
        return frame

    def next_frame(self):
        frame = super(DirectoryFrameReader, self).next_frame()
        self.__frame_index += 1
        return frame

    def has_next(self):
        '''
        Return False when all images have been read
        '''
        return self.__frame_index < len(self.__image_files)

    def release(self):
        '''
        Release all image file
        '''
        self.__frame_index = 0
        self.__image_files = []


if __name__ == '__main__':
    import doctest
    doctest.testmod()
