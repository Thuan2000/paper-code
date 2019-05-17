import time
from core import face_detector, tf_graph
from pipe import worker, task
from pipe.trace_back import process_traceback
from utils.logger import logger
from config import Config


class FaceDetectWorker(worker.Worker):
    '''
    Detect face from frame stage
    Input: frame, frame_id
    Output: bbs (bouncing boxes), pts (landmarks)
    # 798.5 MiB in GPU Ram
    '''

    def doInit(self):
        face_graph = tf_graph.FaceGraph()
        try:
            self.face_detector = face_detector.MTCNNDetector(face_graph, scale_factor=Config.MTCNN.SCALE_FACTOR)
        except:
            logger.exception("CUDA device out of memory")
        super(FaceDetectWorker, self).__init__()
        self.face_count = 0
        self.detected_frame_count = 0
        print(self.name, '=' * 10)

    @process_traceback
    def doFrameTask(self, _task):
        # start = time.time()
        data = _task.depackage()
        frame, frame_info = data['frame'], data['frame_info']

        # timer.detection_start()
        bbs, pts = self.face_detector.detect_face(frame)
        # timer.detection_done()
        # logger.info(
        #     'Frame: %s, bbs: %s, pts: %s' % (frame_info, list(bbs), list(pts)))
        _task = task.Task(task.Task.Face)
        _task.package(bbs=bbs, pts=pts, frame=frame, frame_info=frame_info)
        self.putResult(_task)
        nrof_faces = len(bbs)
        if nrof_faces > 0:
            self.face_count += nrof_faces
            self.detected_frame_count += 1
        # print(self.name, time.time() - start, nrof_faces)

    def doFinish(self):
        logger.info('Face count: %s' % self.face_count)
        logger.info('Frame with face: %s' % self.detected_frame_count)
