import time
import numpy as np
import cv2
from pipe import worker, task
import prod.unilever_alert.config as Config
from prod.unilever_alert.worker import corruption_checking


class ViolationAlertWorker(worker.Worker):
    '''
    sending images of a violation and send to next step,
    not tracking each violations, just send the video if there is violation or not
    '''
    def __init__(self, **kwargs):
        self.points = kwargs.get('points')
        self.alert_type = kwargs.get('alert_type')

    def doInit(self):
        self.alert_images = []
        self.alert_tick = 0
        self.violation_checker = corruption_checking.CorruptionChecker(points=self.points)

    def doEventTask(self, _task):
        data = _task.depackage()
        frame = data['frame']
        frame_info = data['frame_info']
        trackers = data['trackers']

        tlwhs = [tracker.to_tlwh().astype(np.int_) for tracker in trackers]
        # draw violations
        image = np.copy(frame)
        for tlwh in tlwhs:
            if self.violation_checker.line_violation(tlwh):
                tlbr = tlwh
                tlbr[2:] = tlbr[:2] + tlbr[2:]
                image = cv2.rectangle(frame, tuple(tlbr[:2]), tuple(tlbr[2:]), (0,0,255), 2)
                self.alert_tick = 0

        self.alert_images.append(image)
        # in waiting time to add ending buffer
        if len(self.alert_images) > Config.ViolationAlert.BUFFER:
                self.alert_tick += 1
        # in starting buffer
        else:
            trim_idx = min(len(self.alert_images), Config.ViolationAlert.BUFFER)
            self.alert_images[-trim_idx:]

        # Update alert images
        if self.alert_tick > Config.ViolationAlert.BUFFER:
            _task = task.Task(task.Task.Event)
            _task.package(alert_type=self.alert_type, images=self.alert_images)
            self.putResult(_task)
            self.alert_images = []
            self.alert_tick = 0
