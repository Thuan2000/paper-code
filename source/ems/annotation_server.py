import os
import time
import math
import zipfile
import hnswlib
import numpy as np
from collections import Counter
from bson.objectid import ObjectId
from collections import defaultdict
from queue import PriorityQueue
from threading import Thread
from ems import base_server
from core import matcher
from core import frame_reader as frameReader
from core.cv_utils import create_if_not_exist, clear_folder, is_exist
from pipe import pipeline, stage, task
from worker import face_detect_worker, face_detect_worker, face_preprocess_worker, face_extract_worker
from worker import tracking_worker, matching_worker, storage_worker, database_worker
from utils import database
from utils.logger import logger
from config import Config



class AnnotationServer(base_server.AbstractServer):
    SOURCE_STREAM = 'stream'
    SOURCE_CSV = 'csv'
    SOURCE_VIDEO = 'video'
    SOURCE_ZIP = 'zip'

    STATUS_READY = 'ready'
    STATUS_ERROR = 'error'
    STATUS_SUBMITTED = 'submitted'
    STATUS_PROCESSING = 'processing'

    def __init__(self, subcription_id):
        self.subcription_id = subcription_id
        self.database = database.AnnotationDatabase()
        self.dataset = self.database.mongodb_dataset.find_one({
            'subscription':
            ObjectId(self.subcription_id)
        })
        self.dataset_id = str(self.dataset.get('_id'))
        self.database.current_processing_dataset(self.dataset_id)

        # TODO: Matcher for this annotations server, consider to
        # break matcher base on segment to handle very large dataset
        self.matcher_path = os.path.join(Config.Dir.MATCHER_DIR,
                                         '%s.pkl' % self.dataset_id)
        self.matcher = matcher.KdTreeMatcher()
        self.model = Config.Model.FACENET_DIR
        super(AnnotationServer, self).__init__()

    def init(self):
        self.create_dir()
        self.prepare_matcher()
        self.do_process_dataset()

    def add_endpoint(self):
        self.app.add_url_rule('/reprocess', 'reprocess', self.reprocess_api, methods=['POST'])
        self.app.add_url_rule('/findNearest', 'findNearest', self.find_nearest_api, methods=['POST'])
        self.app.add_url_rule('/delete', 'delete', self.delete_api, methods=['POST'])

    def prepare_matcher(self):
        if is_exist(self.matcher_path):
            self.matcher.load_model(self.matcher_path)
        else:
            self.update_matcher()

    def create_dir(self):
        create_if_not_exist(Config.Dir.DATASET_DIR)
        create_if_not_exist(Config.Dir.MATCHER_DIR)
        create_if_not_exist(Config.Dir.ANNOTATION_DIR)
        create_if_not_exist(Config.Dir.LOG_DIR)

    def do_process_dataset(self):
        source_type = self.dataset.get('sourceType', AnnotationServer.SOURCE_CSV)
        dataset_id = self.dataset['_id']
        status = self.dataset['status']

        if status != AnnotationServer.STATUS_READY:
            _id = ObjectId(self.dataset_id)
            self.database \
                .mongodb_dataset \
                .update({'_id': _id}, \
                        {'$set': {'status': AnnotationServer.STATUS_PROCESSING}})

            if source_type != AnnotationServer.SOURCE_CSV:
                if source_type == AnnotationServer.SOURCE_VIDEO:
                    frame_reader = self.get_video_frame_reader()
                elif source_type == AnnotationServer.SOURCE_ZIP:
                    frame_reader = self.get_zip_file_frame_reader()

                if frame_reader is not None:
                    t = Thread(target=self.process_stream, kwargs={'frame_reader':frame_reader})
                    t.start()
                    t.join()

            print('=== Update matcher and process find nearest neighbors')
            self.update_matcher()
            self.find_nearest_image_ids()

            if self.database.get_dataset_status() != AnnotationServer.STATUS_ERROR:
                self.database \
                    .mongodb_dataset \
                    .update({'_id': _id},
                            {'$set': {'status': AnnotationServer.STATUS_READY}})

    # stream_url can be both video or folder
    def validate_stream_url(self, stream_url):
        _id = ObjectId(self.dataset_id)
        if not is_exist(stream_url):
            self.database.mongodb_dataset \
                    .update({'_id': _id}, \
                            {'$set': {'status': AnnotationServer.STATUS_ERROR, \
                                      'reason': 'Video file not found'}})
            return False
        return True

    def get_video_frame_reader(self):
        video_url = self.dataset['storagePath']
        video_path = os.path.join(Config.Dir.VIDEO_DIR, video_url)
        if self.validate_stream_url(video_path):
            frame_reader = frameReader.URLFrameReader(video_path)
            return frame_reader
        return None


    def get_zip_file_frame_reader(self):
        # zipfile create sub folder so cant read directly from zip_file_abs_path
        zip_file_url = self.dataset['storagePath']
        zip_file_abs_path = os.path.join(Config.Dir.VIDEO_DIR, zip_file_url)
        folder_path = os.path.splitext(zip_file_abs_path)[0]
        print('unzipping files')
        with zipfile.ZipFile(zip_file_abs_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        if self.validate_stream_url(folder_path):
            frame_reader = frameReader.DirectoryFrameReader(folder_path)
            return frame_reader
        return None

    def process_stream(self, frame_reader):
        '''
        This is main function
        '''
        _id = ObjectId(self.dataset_id)
        _pipeline = self.build_pipeline()

        print('Begin')
        frame_counter = 0
        while frame_reader.has_next():
            print('process frame number_ ', frame_counter)
            # logger.info('process frame number_ ', frame_counter)
            frame = frame_reader.next_frame()
            if frame is None:
                break

            print('Read frame', frame_counter, frame.shape)
            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                _task = task.Task(task.Task.Frame)
                _task.package(frame=frame, frame_info=frame_counter)
                _pipeline.put(_task)

            frame_counter += 1

        print("Wait for executor to finish it jobs")
        _pipeline.put(None)
        frame_reader.release()

        return True

    def clear_data_dir(self):
        dataset_path = os.path.join(Config.Dir.DATASET_DIR, self.dataset_id)
        annotation_path = os.path.join(Config.Dir.ANNOTATION_DIR,
                                       self.dataset_id)
        self.database.mongodb_image.remove({
            'dataset': ObjectId(self.dataset_id)
        })
        clear_folder(dataset_path)
        clear_folder(annotation_path)

    def create_data_dir(self):
        dataset_path = os.path.join(Config.Dir.DATASET_DIR, self.dataset_id)
        annotaion_path = os.path.join(Config.Dir.ANNOTATION_DIR,
                                      self.dataset_id)
        create_if_not_exist(dataset_path)
        create_if_not_exist(annotaion_path)

    def build_pipeline(self):
        stageDetectFace = stage.Stage(face_detect_worker.FaceDetectWorker, 1)
        stagePreprocess = stage.Stage(face_preprocess_worker.PreprocessDetectedFaceWorker, 1)
        stageExtract = stage.Stage(face_extract_worker.MultiFacesExtractWorker, 1)
        stageTrack = stage.Stage(tracking_worker.FullTrackTrackingWorker, 1, area='Anno')
        stageMatching = stage.Stage(
            matching_worker.MatchingWorker,
            1,
            database=self.database,
            matcher=self.matcher,
            area='Anno')
        stageStorage = stage.Stage(
            storage_worker.AnnotationStorageWorker, 1, dataset_id=self.dataset_id)
        stageDatabase = stage.Stage(
            database_worker.AnnotationDatabaseWorker,
            1,
            dataset_id=self.dataset_id,
            database=self.database)

        stageDetectFace.link(stagePreprocess)
        stagePreprocess.link(stageExtract)
        stageExtract.link(stageTrack)
        stageTrack.link(stageMatching)
        stageMatching.link(stageStorage)
        stageMatching.link(stageDatabase)

        _pipeline = pipeline.Pipeline(stageDetectFace)
        return _pipeline

    def update_matcher(self):
        print('Matcher updated')
        self.matcher.build(self.database)
        self.matcher.save_model(self.matcher_path)

    def reprocess_api(self):
        logger.debug('Reprocess dataset: %s' % self.dataset_id)
        self.database \
                .mongodb_dataset \
                .update({'_id': self.dataset['_id']}, \
                        {'$set': {'status': AnnotationServer.STATUS_PROCESSING}})
        self.do_process_dataset()
        return AnnotationServer.response_success(result='Processing dataset')

    def find_nearest_api(self):
        image_id = self.request.json.get('image_id')
        logger.debug('FindNearest, image_id: %s' % image_id)
        try:
            result = self.database.nearest_image_ids_for_image_id(image_id)
            logger.debug(result)
            return self.response_success(result=result)
        except Exception as e:
            logger.error(e, exec_info=True)
            return self.response_error('Find nearest failed')

    def delete_api(self):
        self.clear_data_dir()

    def find_nearest_image_ids(self):
        # start = time.time()
        # Get all images and find theirs nearest neighbor
        # first we need to index the algo

        doc_ids, embs = self.database.get_ids_and_embs()
        if embs.any():
            nrof_images, dim = embs.shape
            k = min(Config.FindNearest.NROF_K, nrof_images)
            doc_ids_idxs = np.arange(nrof_images)
            # Declaring index
            p = hnswlib.Index(space='l2', dim=dim) # possible options are l2, cosine or ip
            # Initing index - the maximum number of elements should be known beforehand
            p.init_index(max_elements=nrof_images, ef_construction=200, M=16)
            # # Element insertion (can be called several times):
            p.add_items(embs, doc_ids_idxs)
            # # Controlling the recall by setting ef:
            p.set_ef(k+10) # ef should always be > k

            for idx, doc_id in enumerate(doc_ids):
                emb = embs[idx]
                predict_idxs, _ = p.knn_query(emb, k=k)
                predict_ids = [doc_ids[_idx] for _idx in predict_idxs[0]]
                self.database.update_field_by_ObjectId(doc_id, nearestIds=predict_ids)
                print('======= Update nearest neighbors for %s' % (doc_id))
        else:
            print('Dont have any images to process')
