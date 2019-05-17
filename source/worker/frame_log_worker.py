import json
from collections import defaultdict
from core.tracking import tracker_results_dict
from pipe import worker, task
from pipe.trace_back import process_traceback
from config import Config


class FrameLoggerWorker(worker.Worker):

    def __init__(self, **args):
        self.log_file = args.get('log_file')
        super(FrameLoggerWorker, self).__init__()

    def doInit(self):
        self.track_results = tracker_results_dict.TrackerResultsDict()

    @process_traceback
    def doTask(self, _task):
        # task_name, package = _task
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        deleted_trackers = data['trackers']
        trackers_return_dict = tracker_results_dict.TrackerResultsDict()
        for tracker in deleted_trackers:

            tracker_return_dict = tracker_results_dict.TrackerResultsDict()
            for element in tracker.elements:
                tracker_return_dict.update(element.frame_id, tracker.face_id,
                                           element.bounding_box)
            trackers_return_dict.merge(tracker_return_dict)
        self.track_results.merge(trackers_return_dict)

    def doFinish(self):
        tracker_results_dict = self.track_results.tracker_results_dict
        for frame_id in tracker_results_dict.keys():
            tracker_result = tracker_results_dict[frame_id]
            for i, name in enumerate(tracker_result.track_names):
                bb0 = int(tracker_result.bounding_boxes[i][0])
                bb1 = int(tracker_result.bounding_boxes[i][1])
                bb2 = int(tracker_result.bounding_boxes[i][2])
                bb3 = int(tracker_result.bounding_boxes[i][3])
                bb = [bb0, bb1, bb2, bb3]
                # result[frame_id][name] = bb
                tracker_result[frame_id][name] = bb
        f = open(self.log_file, 'w')
        # json.dump(result, f)
        json.dump(tracker_results_dict, f)
        f.close()
