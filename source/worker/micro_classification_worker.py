import os
import requests
import numpy as np
from collections import Counter
from pipe import worker, task
from utils import simple_request
from utils.logger import logger
from config import Config


class MassanCustomerClassificationWorker(worker.Worker):
    '''
    this worker currently send relative path of image file since api and server is share
    the same physical disk
    '''

    def __init__(self, **kwargs):
        url = kwargs.get('url')
        area = kwargs.get('area')
        volume_name = kwargs.get('volume_name')
        path_prefix = os.path.join(volume_name, '_data', area)
        self.requester = simple_request.TrackerHTTPRequest(url, path_prefix=path_prefix)

    def doFaceTask(self, _task):
        data = _task.depackage()
        tracker = data['tracker']
        # is_ignored: to determine if the record is showed or not
        tracker.is_ignored = True
        image_paths = []
        for element in tracker.elements:
            image_path = os.path.join(str(tracker.track_id), element.image_id + '.jpg')
            image_paths.append(image_path)

        classified_result = \
            self.requester.post_list(Config.MicroServices.IMAGES, image_paths)
        if (classified_result is not None) and ('predictions' in classified_result):
            predictions = classified_result['predictions']
            count = Counter(predictions).most_common(1)
            prediction, _ = count[0]
            print(self.name, 'predict %s: %s' % (tracker.track_id, prediction))
            if prediction == True:
                tracker.is_ignored = False

        _task = task.Task(task.Task.Face)
        _task.package(type=Config.Worker.TASK_TRACKER, tracker=tracker)
        self.putResult(_task)


class AgeGenderPredictionWorker(worker.Worker):

    def __init__(self, **kwargs):
        url = kwargs.get('url')
        area = kwargs.get('area')
        volume_name = kwargs.get('volume_name')
        path_prefix = os.path.join(volume_name, '_data', area)
        self.requester = simple_request.TrackerHTTPRequest(url, path_prefix=path_prefix)

    def doFaceTask(self, _task):
        data = _task.depackage()
        tracker = data['tracker']

        gender, age = None, None
        request_data = []
        for element in tracker.elements:
            image_path = os.path.join(str(tracker.track_id), element.image_id + '.jpg')
            landmarks = element.landmarks.astype(np.int)
            landmarks_str = '_'.join(list(map(str, landmarks)))
            _data = ','.join([image_path, landmarks_str])
            request_data.append(_data)

        predictions = self.requester.post_list(Config.MicroServices.IMAGES, request_data)
        if predictions is not None and ('predictions' in predictions):
            result_list = predictions['predictions']
            genders = []
            ages = []
            for gender, age in result_list:
                if gender is not None:
                    genders.append(gender)
                if age is not None:
                    ages.append(age)

            if genders and ages:
                gender, _ = Counter(genders).most_common(1)[0]
                gender = 'Male' if gender == 1 else 'Female'
                age = int(np.mean(ages))

        print('Prediction for tracker %s: age-%s, gender-%s' % (tracker.track_id, age, gender))
        # implicit attribute asign for tracker age and gender
        tracker.age = age
        tracker.gender = gender

        _task = task.Task(task.Task.Face)
        _task.package(type=Config.Worker.TASK_TRACKER, tracker=tracker)
        self.putResult(_task)

