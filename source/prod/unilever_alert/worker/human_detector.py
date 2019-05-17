import time
from pipe import worker, task
from utils.logger import logger
from core.tracking import detection
from core import background_subtraction
import prod.unilever_alert.config as Config


class HumanDetectWorker(worker.Worker):

    def doInit(self):
        # TODO: add this
        self.detector = background_subtraction.BGSProcess(history=Config.MOG.HISTORY,
            backgroundratio=Config.MOG.BACKGROUNDRATIO,
            shadowthresh=Config.MOG.SHADOWTHRESH,
            object_size=Config.MOG.OBJECT_SIZE,
            detectshadow_flag=Config.MOG.DETECTSHADOW_FLAG,
            varthreshold=Config.MOG.VARTHRESHOLD,
            rect_size=Config.MOG.RECT_SIZE,
            ellipse_size=Config.MOG.ELLIPSE_SIZE,
            high_rect_size=Config.MOG.HIGH_RECT_SIZE)

    def doFrameTask(self, _task):
        data = _task.depackage()
        frame = data['frame']
        frame_info = data['frame_info']

        bboxes = self.detector.process(frame)
        detections = [detection.Detection(bbox, 100, 100) for bbox in bboxes]
        if len(detections) > 0:
            _task = task.Task(task.Task.Frame)
            _task.package(frame=frame, frame_info=frame_info,
                         detections=detections)
            self.putResult(_task)
            logger.debug('Found %s detection' % len(detections))

