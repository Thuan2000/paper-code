import time
import numpy as np
from pipe import worker, task
from pipe.trace_back import process_traceback
from core import pedestrian_extractor
from core.cv_utils import CropperUtils
from core.tracking import detection
from utils.logger import logger
from config import Config

class PedestrianExtractWorker(worker.Worker):

    # 936.0 MiB
    def doInit(self, model_filename=Config.Model.MARS_DIR, batch_size=1):
        try:
            self.encoder = pedestrian_extractor.create_box_encoder(model_filename, batch_size=batch_size)
        except:
            logger.exception("CUDA out off memory", exc_info=True)
        print(self.name, '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        bbs, scores, frame, frame_info = data['bbs'], data['scores'], data[
            'frame'], data['frame_info']

        nrof_faces = len(bbs)
        # timer.preprocess_start()
        faces = []
        for i in range(nrof_faces):
            display_face, padded_bb_str = CropperUtils.crop_display_face(
                frame, np.asarray(bbs[i]))
            face = detection.PedestrianInfo(np.asarray(bbs[i]), frame_info, display_face,
                padded_bb_str, scores[i])
            faces.append(face)

        # logger.info("Extract: %s, #Faces: %s" % (frame_info, len(faces)))
        # timer.extract_start()
        embs = self.encoder(frame, bbs_to_tlwhs(faces))
        face_infos = []
        for i, face in enumerate(faces):
            face.embedding = embs[i]
            face_infos.append(face)

        _task = task.Task(task.Task.Face)
        _task.package(faces=face_infos)
        self.putResult(_task)

def bbs_to_tlwhs(pdts):
    tlwhs = []
    for pdt in pdts:
        tlwhs.append(pdt.to_tlwh())
    return tlwhs

