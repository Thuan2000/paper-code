import os
from scipy import misc
from pipe import worker, task
from pipe.trace_back import process_traceback
from core.cv_utils import create_if_not_exist, PickleUtils, CropperUtils
from utils.logger import logger
from config import Config


class DashboardStorageWorker(worker.Worker):

    def __init__(self, **args):
        self.tracking_dir = args.get('tracking_dir,', Config.Dir.TRACKING_DIR)
        create_if_not_exist(self.tracking_dir)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        deleted_trackers = data['trackers']
        for tracker in deleted_trackers:
            track_id_path = os.path.join(self.tracking_dir,
                                         str(tracker.track_id))
            create_if_not_exist(track_id_path)
            for tracker_element in tracker.elements:
                save_image = tracker_element.display_image
                image_id = '{}_{}'.format(tracker.track_id,
                                          tracker_element.str_info())
                image_name = image_id + '.jpg'
                image_path = os.path.join(track_id_path, image_name)
                if os.path.exists:
                    # path exist if have dump in realtime
                    continue
                misc.imsave(image_path, save_image)
                logger.debug('Save image: %s' % image_id)


class AnnotationStorageWorker(worker.Worker):

    def __init__(self, **args):
        self.dataset_dir = os.path.join(Config.Dir.DATASET_DIR,
                                        args.get('dataset_id'))
        self.annotation_dir = os.path.join(Config.Dir.ANNOTATION_DIR,
                                           args.get('dataset_id'))
        create_if_not_exist(self.dataset_dir)
        create_if_not_exist(self.annotation_dir)

    @process_traceback
    def doFaceTask(self, _task):
        data = _task.depackage()
        task_name = data['type']
        if task_name != Config.Worker.TASK_TRACKER:
            return

        tracker = data['tracker']
        if tracker.face_id == Config.Track.INIT_FACE_ID or \
                tracker.face_id == Config.Track.BAD_TRACK:
            tracker.assign_face_id('Anno')

        for tracker_element in tracker.elements:
            face_image = tracker_element.face_image
            cropped_image = CropperUtils.reverse_display_face(tracker_element.face_image,
                                                              tracker_element.str_padded_bbox)
            bbox_confident = tracker_element.bbox_confident
            landmark = tracker_element.landmarks
            image_id = '{}_{}'.format(tracker.track_id, tracker_element.str_info())
            image_name = image_id + '.jpg'
            image_pkl = image_id + '.pkl'
            dataset_pkl_path = os.path.join(self.dataset_dir, image_pkl)
            annotation_image_path = os.path.join(self.annotation_dir, image_name)
            PickleUtils.save_pickle(dataset_pkl_path, value=(cropped_image, tracker_element.embedding,
                                                                bbox_confident, landmark))
            misc.imsave(annotation_image_path, face_image)
            logger.debug('Save image: %s' % image_id)
