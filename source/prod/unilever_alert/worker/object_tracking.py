import time
from pipe import worker, task
from core.tracking import tracker_manager
from utils.logger import logger
import prod.unilever_alert.config as Config

class TrackingWorker(worker.Worker):

    def doInit(self):
        self.tracker_manager = tracker_manager.ObjectTrackerManager()

    def doFrameTask(self, _task):
        data = _task.depackage()
        frame = data['frame']
        frame_info = data['frame_info']
        detections = data['detections']

        self.tracker_manager.predict()
        self.tracker_manager.update(detections)
        trackers = self.tracker_manager.get_confirmed_trackers()

        if len(trackers) > 0:
            _task = task.Task(task.Task.Event)
            _task.package(frame=frame, frame_info=frame_info,
                         trackers=trackers)
            self.putResult(_task)
            # logger.info('sending %s object to check violation' % len(trackers))
