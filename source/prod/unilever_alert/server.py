from threading import Thread
import queue
import time
import os
import numpy as np
import cv2
import psutil
from ems import base_server
from pipe import stage, task, pipeline
from core import frame_reader, frame_queue
from core.cv_utils import create_if_not_exist
from prod.unilever_alert.worker import database, object_tracking, messenger, storage
from prod.unilever_alert.worker import violation_alert, human_detector
from utils.logger import logger
from config import Config


class CameraStatus:
    RUNNING = 'running'
    STOPPED = 'stopped'
    ERROR = 'error'


class UnileverServer(base_server.AbstractServer):

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.database = database.Database(camera_id)
        super(UnileverServer, self).__init__()

    def init(self):
        rtsp = self.database.get_rtsp_link()
        # rtsp = '/iq_facial_recognition/source/prod/unilever_alert/Timeline_1.mp4'
        print(rtsp)
        self.frame_reader = frame_reader.URLFrameReader(rtsp, convert_color=False, re_source=True)
        self.frame_queue = frame_queue.FrameQueue(self.frame_reader)
        self.points = [(578, 93), (853, 509)]

        self.messenger = messenger.SocketMessenger(Config.Socket.HOST, Config.Socket.PORT)
        self.snapshot_cache = queue.Queue(maxsize=Config.ViolationAlert.NROF_SNAPSHOT)
        self.statistic_queue = queue.Queue()

        self.rel_snapshot_dir = '%s/%s' % (Config.Dir.SNAPSHOT, self.camera_id)
        abs_snapshot_dir = '%s/%s' % (Config.Dir.DATA_DIR, self.rel_snapshot_dir)
        snapshot_dir = '%s/%s' % (Config.Dir.DATA_DIR, Config.Dir.SNAPSHOT)
        create_if_not_exist(Config.Dir.DATA_DIR)
        create_if_not_exist(snapshot_dir)
        create_if_not_exist(abs_snapshot_dir)

        stat_thread = Thread(target=self.statistic_collector)
        process_thread = Thread(target=self.do_process)
        stat_thread.start()
        process_thread.start()

    def add_endpoint(self):
        self.app.add_url_rule(
            '/getSnapshot', 'getSnapshot', self.snapshot_api, methods=['GET'])

    def do_process(self):
        self.database.set_camera_status(CameraStatus.RUNNING)
        _pipeline = self.build_pipeline()
        self.frame_queue.start()
        frame_counter = 0

        while True:
            frame = self.frame_queue.next_frame()
            if frame is not None:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_counter)
                _pipeline.put(_task)
                status, data = _pipeline.get(timeout=1e-6)
                if status:
                    print(data)
                    self.messenger.send_alert(**data)
                self.add_queue(frame)
                frame_counter += 1

            if not self.statistic_queue.empty():
                stats = self.statistic_queue.get()
                print(stats)
                self.messenger.send_statistic(**stats)

    def snapshot_api(self):
        snapshots = self.get_all_queue()
        urls = []
        # Save snapshots to disk then send urls
        for snapshot in snapshots:
            img_rel_path = '%s/%s.jpg' % (self.rel_snapshot_dir, time.time())
            img_abs_path = '%s/%s' % (Config.Dir.DATA_DIR, img_rel_path)
            cv2.imwrite(img_abs_path, snapshot)
            urls.append(img_rel_path)
        if urls:
            return self.response_success({'images':urls})
        else:
            return self.response_error('Snapshot cache is empty')

    def add_queue(self, frame):
        if self.snapshot_cache.full():
            self.snapshot_cache.get()
        self.snapshot_cache.put(frame)

    def get_all_queue(self):
        snapshots = []
        while not self.snapshot_cache.empty():
            snapshots.append(self.snapshot_cache.get())
        return snapshots

    def build_pipeline(self):
        logger.info('Getting camera infos')
        fps, w, h = self.frame_reader.get_info()
        stageDetect = stage.Stage(human_detector.HumanDetectWorker)
        stageTrack = stage.Stage(object_tracking.TrackingWorker)
        stageAlert = stage.Stage(violation_alert.ViolationAlertWorker,
                                      points=self.points, alert_type=Config.Violations.TRESPASSING)
        stageStorage = stage.Stage(storage.StorageWorker, fps=fps,
                                        video_dim=(w,h), camera_id=self.camera_id)
        stageDatabase = stage.Stage(database.DatabaseWorker, database=self.database)

        stageDetect.link(stageTrack)
        stageTrack.link(stageAlert)
        stageAlert.link(stageStorage)
        stageStorage.link(stageDatabase)

        _pipeline = pipeline.Pipeline(stageDetect)
        return _pipeline

    def statistic_collector(self):
        pid = os.getpid()
        cpu_percent = psutil.Process(pid).cpu_percent()
        rss, vms = psutil.Process(pid).memory_info()[:2]

        # create return data
        stats = {'cpu': {'used_percent': cpu_percent},
                 'ram': {'rss': rss, 'vms': vms},
                 'gpu': {}}

        self.statistic_queue.put(stats)
        time.sleep(5)
