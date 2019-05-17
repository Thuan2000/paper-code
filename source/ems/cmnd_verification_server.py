from core import face_detector, tf_graph
from core import matcher, frame_reader
from worker import face_detect_worker, face_extract_worker, face_preprocess_worker
from pipe import pipeline, task, stage

class CmndVerificationServer():
    def __init__(self):
        # Use a new frame_reader
        # todo: integrate a special frame reader for this case. 
        self.frame_reader = frame_reader.CMNDFrameReader(self.database)
    def add_endpoint(self):
        pass
    def build_pipeline(self):
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_preprocess_worker.PreprocessDetectedWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.MultiArcFacesExtractWorker, 1)
    def run(self):
        _pipeline = self.build_pipeline()
        frame_number = 0
        while True: 
            frame = self.frame_reader.next_frame()
            if frame is not None: 
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_time)
                frame_number += 1
                _pipeline.put(_task)