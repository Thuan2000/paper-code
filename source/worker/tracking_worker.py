import time
from pipe import worker, task
from pipe.trace_back import process_traceback
from core.tracking import tracker_manager, nn_matching
from utils import timer
from utils.logger import logger
from config import Config


class RealTimeTrackingWorker(worker.Worker):
    '''
    Do tracking for each face in frame
    '''

    def __init__(self, **args):
        self.area = args.get('area', 'TCH')

        self.current_id = args.get('init_tracker_id', 0)
        super(RealTimeTrackingWorker, self).__init__()

    def doInit(self):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", Config.Track.MAX_COSINE_DISTANCE, Config.Track.NN_BUDGET)
        self.tracker_manager = tracker_manager.FaceTrackerManager(
            metric=metric, current_id=self.current_id)
        self.stat_collector = timer.TimerCollector()
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        faces = data['faces']
        self.handleDeletedTrackers(faces)
        self.handleEarlyQualifiedTrackers()

    def handleDeletedTrackers(self, faces):
        self.tracker_manager.predict()
        qualified_trackers, _ = self.tracker_manager.update(faces)
        for tracker in qualified_trackers:
            _task = task.Task(task.Task.Face)
            _task.package(tracker=tracker)
            self.putResult(_task)

    def handleEarlyQualifiedTrackers(self):
        early_qualified_trackers = self.tracker_manager.get_early_qualified_trackers()
        for tracker in early_qualified_trackers:
            _task = task.Task(task.Task.Face)
            _task.package(tracker=tracker)
            self.putResult(_task)

    # def cleanUpTrackers(self):
    #     trackers = self.tracker_manager.cleanup_overtime_tracker()
    #     # currently we only handle qualified tracker only
    #     for tracker in trackers:
    #         if tracker.is_qualified_to_be_recognized() or tracker.is_recognized():
    #             _task = task.Task(task.Task.Face)
    #             _task.package(type=Config.Worker.TASK_EXTRACTION, tracker=tracker)
    #             self.putResult(_task)

    def doFinish(self):
        self.stat_collector.statistic()


class FullTrackTrackingWorker(worker.Worker):
    '''
    Do tracking for each face in frame
    '''

    def __init__(self, **args):
        self.area = args.get('area', 'TCH')

        self.current_id = args.get('init_tracker_id', 0)
        super(FullTrackTrackingWorker, self).__init__()

    def doInit(self):
        metric =nn_matching.NearestNeighborDistanceMetric("cosine", Config.Track.MAX_COSINE_DISTANCE, Config.Track.NN_BUDGET)
        self.tracker_manager = tracker_manager.FaceTrackerManager(metric=metric, current_id=self.current_id)
        self.stat_collector = timer.TimerCollector()
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        faces = data['faces']
        self.handleNewFaces(faces)

    def handleNewFaces(self, faces):
        self.tracker_manager.predict()
        qualified_trackers, _ = self.tracker_manager.update(faces)
        for tracker in qualified_trackers:
            _task = task.Task(task.Task.Face)
            _task.package(tracker=tracker)
            self.putResult(_task)

    def doFinish(self):
        self.stat_collector.statistic()
