import time
import cv2
from pipe import worker, task
from core.cv_utils import create_if_not_exist
from utils.logger import logger
import prod.unilever_alert.config as Config


class StorageWorker(worker.Worker):

    def __init__(self, **kwargs):
        self.fps = kwargs.get('fps')
        self.video_dim = kwargs.get('video_dim')
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.camera_id = kwargs.get('camera_id')
        self.rel_video_dir = '%s/%s' % (Config.Dir.VIDEO, self.camera_id)
        abs_video_dir = '%s/%s' % (Config.Dir.DATA_DIR, self.rel_video_dir)
        create_if_not_exist(Config.Dir.VIDEO_DIR)
        create_if_not_exist(abs_video_dir)

    def doInit(self):
        # TODO: add this
        pass

    def doEventTask(self, _task):
        data = _task.depackage()
        images = data['images']
        alert_type = data['alert_type']

        video_rel_path = '%s/%s.mp4' % (self.rel_video_dir, time.time())
        video_abs_path = "%s/%s" % (Config.Dir.DATA_DIR, video_rel_path)
        writer = cv2.VideoWriter(video_abs_path, self.fourcc, self.fps, self.video_dim)
        for image in images:
            writer.write(image)
        writer.release()

        _task = task.Task(task.Task.Event)
        _task.package(alert_type=alert_type, video_name=video_rel_path)
        self.putResult(_task)

        logger.info('Found violation, saved in %s' % video_rel_path)
