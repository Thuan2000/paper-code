import os
import cv2
from pipe import pipeline, stage, task
from worker import pedestrian_detect_worker, pedestrian_extract_worker
from worker import tracking_worker, micro_classification_worker, database_worker
from core  import matcher, frame_reader
from core.cv_utils import get_biggest_folder_number
from utils import database, socket_client
from config import Config


class RetentionServer():

    def __init__(self):
        self.volume_name = os.environ.get('VOLUME_NAME', '')
        self.database = database.RetentionDashboardDatabase()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.socket = socket_client.RetentionDashboardBlacklistSocket(hostname=Config.Socket.HOST, port=Config.Socket.PORT)
        self.frame_reader = frame_reader.NASFrameReader(self.database, **Config.Source.NAS_INPUT)

    def build_pipeline(self):
        init_tracker_id = get_biggest_folder_number(Config.Dir.TRACKING_DIR) + 1
        stageDetectFace = stage.Stage(pedestrian_detect_worker.PedestrianDetectWorker, 1)
        stageExtract = stage.Stage(pedestrian_extract_worker.PedestrianExtractWorker, 1)
        stageTracking = stage.Stage(tracking_worker.FullTrackTrackingWorker, 1,
            area=Config.Source.AREA, init_tracker_id=init_tracker_id)
        stageClassification = stage.Stage(
            micro_classification_worker.MassanCustomerClassificationWorker, 1,
            area=Config.Source.AREA,
            url=Config.MicroServices.SERVICE_MASAN_CUSTOMER_CLASSIFICATION,
            volume_name=self.volume_name)
        stageDatabase = stage.Stage(database_worker.RetentionDashboardUpdateIsIgnore, 1,
            database=self.database, socket=self.socket)

        stageDetectFace.link(stageExtract)
        stageExtract.link(stageTracking)
        stageTracking.link(stageClassification)
        stageClassification.link(stageDatabase)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline

    def run(self):
        _pipeline = self.build_pipeline()
        frame_number = 0
        while True:
            frame, frame_time  = self.frame_reader.next_frame()
            if frame is not None:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_time)
                frame_number += 1
                _pipeline.put(_task)
