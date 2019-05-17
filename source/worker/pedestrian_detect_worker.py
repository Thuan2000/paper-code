import cv2
import time
import numpy as np
from PIL import Image
from pipe import worker, task
from pipe.trace_back import process_traceback
from core import pedestrian_detector
from config import Config
from utils.logger import logger
from core import background_subtraction

class PedestrianDetectWorker(worker.Worker):
    '''
    Detect face from frame stage
    Input: frame, frame_id
    Output: bbs (bouncing boxes), pts (landmarks)
    # 798.5 MiB in GPU Ram
    '''

    def doInit(self):
        try:
            self.pedestrian_detector = pedestrian_detector.YOLODetector()
        except:
            logger.exception("CUDA device out of memory")
        super(PedestrianDetectWorker, self).__init__()
        roi_cordinate = Config.ROI.ROI_CORDINATE[Config.ROI.USE]
        self.roi_cordinate_np = self.ConvertCordinates(roi_cordinate)
        self.rect = cv2.boundingRect(self.roi_cordinate_np)
        self.roi_cordinate_np_scale = self.roi_cordinate_np - np.array([self.rect[0],self.rect[1]])
        self.centroids = (int(self.rect[2]/2), int(self.rect[3]/2))
        self.pedestrian_count = 0
        self.detected_frame_count = 0
        self.background_subtraction = background_subtraction.BGSProcess()
        print(self.name, '=' * 10)

    def ConvertCordinates(self, roi_cor):
        roi_cor_np = [list(i) for i in roi_cor]
        roi_cor_np = np.array(roi_cor_np, np.int32)
        return roi_cor_np

    def convert(self, frame, roi_cor_np):
        roi = frame[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2]]
        black_mask = np.zeros(roi.shape[:2], dtype = np.int8)
        cv2.fillConvexPoly(black_mask, roi_cor_np, 255)
        roi_mask = cv2.bitwise_and(roi, roi, mask = black_mask)
        return roi_mask, roi

    @process_traceback
    def doFrameTask(self, _task):
        start = time.time()
        data = _task.depackage()
        frame, frame_info = data['frame'], data['frame_info']
        frame = cv2.resize(frame, (1920,1080))
        frame, roi = self.convert(frame, self.roi_cordinate_np_scale)

        bbs = []
        scores = []
        if self.background_subtraction.preprocess(roi):
            pil_im = Image.fromarray(frame)
            bbs, scores = self.pedestrian_detector.detect_image(pil_im)

            for bb in bbs:
                bbx = [(bb[0],bb[1]), (bb[0],bb[3]), (bb[2],bb[3]), (bb[2], bb[1])]
                if (cv2.pointPolygonTest(np.array([bbx], np.int32 ), self.centroids, False) == -1):
                    bbs.remove(bb)

        _task = task.Task(task.Task.Face)
        _task.package(bbs=bbs, scores=scores, frame=roi, frame_info=frame_info)
        self.putResult(_task)

        nrof_faces = len(bbs)
        if nrof_faces > 0:
            self.pedestrian_count += nrof_faces
            self.detected_frame_count += 1

    def doFinish(self):
        logger.info('Pedestrian count: %s' % self.pedestrian_count)
        logger.info('Frame with face: %s' % self.detected_frame_count)
