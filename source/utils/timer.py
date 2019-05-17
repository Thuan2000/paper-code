from datetime import datetime
import collections
import numpy as np
import time


class Timer(object):
    '''
    For tracking how much time have elapse for each stage
    '''

    def __init__(self, frame_id):
        # self.start_time = time.time()
        self.frame_id = frame_id
        self.elapsed_time = {'total': time.time()}
        # self.detection_start_time = 0
        # self.detection_done_time = 0
        # self.preprocess_start_time = 0
        # self.preprocess_done_time = 0
        # self.extract_start_time = 0
        # self.extract_done_time = 0
        # self.track_start_time = 0
        # self.track_done_time = 0
        # self.client_start_time = 0
        # self.client_done_time = 0

    def start(self, time_name):
        self.elapsed_time[time_name] = time.time()

    def done(self, time_name):
        self.elapsed_time[time_name] = time.time(
        ) - self.elapsed_time[time_name]

    def finalize(self):
        self.elapsed_time['total'] = time.time() - self.elapsed_time['total']
        return self.elapsed_time
        # detect_elapsed = self.detection_done_time - self.detection_start_time
        # preprocess_elapsed = self.preprocess_done_time - self.preprocess_start_time
        # extract_elapsed = self.extract_done_time - self.extract_start_time
        # track_elapsed = self.track_done_time - self.track_start_time
        # client_elapsed = self.client_done_time - self.client_start_time
        # return (time_elapsed, detect_elapsed,
        #         preprocess_elapsed, extract_elapsed,
        #         track_elapsed, client_elapsed)

    # def detection_start(self):
    #     self.detection_start_time = time.time()

    # def detection_done(self):
    #     self.detection_done_time = time.time()

    # def preprocess_start(self):
    #     self.preprocess_start_time = time.time()

    # def preprocess_done(self):
    #     self.preprocess_done_time = time.time()

    # def extract_start(self):
    #     self.extract_start_time = time.time()

    # def extract_done(self):
    #     self.extract_done_time = time.time()

    # def track_start(self):
    #     self.track_start_time = time.time()

    # def track_done(self):
    #     self.track_done_time = time.time()

    # def client_start(self):
    #     self.client_start_time = time.time()

    # def client_done(self):
    #     self.client_done_time = time.time()


class TimerCollector(object):
    '''
    Collect timer for each individual face
    Use this at the final stage
    '''

    def __init__(self):
        self.stat_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.stat = collections.defaultdict(lambda: [])
        self.fps = []

    def collect(self, timer):
        time_elapsed, detect_elapsed, \
            preprocess_elapsed, extract_elapsed, \
            track_elapsed, client_elapsed = timer.finalize()

        self.stat['time'].append(time_elapsed)
        self.stat['detect'].append(detect_elapsed)
        self.stat['preprocess'].append(preprocess_elapsed)
        self.stat['extract'].append(extract_elapsed)
        self.stat['track'].append(track_elapsed)
        self.stat['client'].append(client_elapsed)

    def collect_fps(self, start_time):
        fps = 1 / (time.time() - start_time)
        print('FPS', fps)
        self.fps.append(fps)

    def statistic(self):
        print("Stat name", self.stat_name)
        print("||" * 40)
        for stat_name, values in self.stat.items():
            txt = 'Stat: %s, \tMax: %.4f \tMin: %.4f \tAverage: %.4f' % \
                  (stat_name, np.max(values), np.min(values), np.average(values))
            print(txt)

        print("||" * 40)
