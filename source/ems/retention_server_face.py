import os
import cv2
from core import matcher, frame_reader
from core.cv_utils import get_biggest_folder_number
from pipe import pipeline, stage, task
from worker import face_detect_worker, face_preprocess_worker, face_extract_worker
from worker import tracking_worker, matching_worker, database_worker, micro_classification_worker
from utils import database, socket_client
from config import Config


class RetentionServer():

    def __init__(self):
        self.volume_name = os.environ.get('VOLUME_NAME', '')
        self.database = database.RetentionDashboardDatabase()
        self.matcher = matcher.HNSWMatcher()
        self.matcher.build(self.database)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.frame_reader = frame_reader.NASFrameReader(self.database, **Config.Source.NAS_INPUT)
        self.socket = socket_client.RetentionDashboardSocket(hostname=Config.Socket.HOST, port=Config.Socket.PORT)

    def build_pipeline(self):
        init_tracker_id = get_biggest_folder_number(Config.Dir.TRACKING_DIR) + 1
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_preprocess_worker.PreprocessDetectedArcFaceWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.MultiArcFacesExtractWorker, 1)
        stageTracking = stage.Stage(tracking_worker.RealTimeTrackingWorker, 1,
            area=Config.Source.AREA, init_tracker_id=init_tracker_id)
        stageMatching = stage.Stage(
            matching_worker.MatchingWorker,
            size=1,
            database=self.database,
            matcher=self.matcher,
            socket=self.socket,
            area=Config.Source.AREA)
        stageAgeGenderPrediction = stage.Stage(
            micro_classification_worker.AgeGenderPredictionWorker, 1,
            area=Config.Source.AREA,
            url=Config.MicroServices.SERVICE_AGE_GENDER_PREDICTION,
            volume_name=self.volume_name)
        stageDatabase = stage.Stage(database_worker.RetentionDashboardDatabaseWorker, 1,
            database=self.database, socket=self.socket)

        stageDetectFace.link(stagePreprocess)
        stagePreprocess.link(stageExtract)
        stageExtract.link(stageTracking)
        stageTracking.link(stageMatching)
        stageMatching.link(stageAgeGenderPrediction)
        stageAgeGenderPrediction.link(stageDatabase)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline

    def run(self):
        _pipeline = self.build_pipeline()
        frame_number = 0
        while True:
            frame, frame_time = self.frame_reader.next_frame()
            if frame is not None:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_time)
                frame_number += 1
                _pipeline.put(_task)
