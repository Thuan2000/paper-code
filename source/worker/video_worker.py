from pipe import worker
from core.tracking import tracker_results_dict
from pipe.trace_back import process_traceback
from config import Config


class VideoWriterWorker(worker.Worker):

    def __init__(self, **args):
        self.video_out = args.get('video_out')
        self.database = args.get('database')
        super(VideoWriterWorker, self).__init__()

    def doInit(self):
        self.track_results = tracker_results_dict.TrackerResultsDict()

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:  # or \
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
        self.video_out.write_track_video(
            self.track_results.tracker_results_dict, self.database)
        self.video_out.release_out()
