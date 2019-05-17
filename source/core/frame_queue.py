#!/usr/bin/python
# -*- coding: utf-8 -*-
import threading
from scipy import misc
import os
import cv2
import shutil
import gc
import time
from utils.logger import logger
from core.cv_utils import create_if_not_exist
from config import Config

class FrameQueue(threading.Thread):

    def __init__(self,
                 frame_reader,
                 thread_lock=threading.Lock(),
                 max_queue_size=Config.Frame.FRAME_QUEUE_SIZE,
                 max_frame_on_disk=Config.Frame.MAX_FRAME_ON_DISK,
                 on_disk_save_path=Config.Frame.FRAME_ON_DISK):

        threading.Thread.__init__(self)

        self.frame_reader = frame_reader
        self.max_queue_size = max_queue_size
        self.max_frame_on_disk = max_frame_on_disk
        self.thread_lock = thread_lock
        self.stream_time = time.time()

        self.queue = []
        self.on_disk_save_path = on_disk_save_path
        self.queue_on_disk = []
        self.frame_number = 0
        # if queue on disk is just full, need to process all the images before running again
        self.queue_on_disk_is_just_full = False
        self.running = True

        # remove old local cache
        self.clear_queue_on_disk()
        create_if_not_exist(self.on_disk_save_path)

    def run(self):
        self.running = True
        while self.running:
            if self.can_write_to_queue():
                self.load_frame()

    def load_frame(self):
        frame = self.frame_reader.next_frame()
        if frame is not None:
            with self.thread_lock:
                if len(self.queue) <= self.max_queue_size:
                    self.queue.append(frame)
                    gc.collect()
                else:
                    file_name = os.path.join(self.on_disk_save_path,
                                             str(self.frame_number) + '.jpg')
                    cv2.imwrite(file_name, frame)
                    self.frame_number += 1
                    self.queue_on_disk.append(file_name)
                    del frame

    def next_frame(self):
        frame = None
        is_ram_queue_full = len(self.queue) >= self.max_queue_size
        with self.thread_lock:
            if len(self.queue) > 0:
                frame = self.queue.pop(0)
            else:
                self.queue.clear()

            if is_ram_queue_full:
                if len(self.queue_on_disk) > 0:
                    image_path = self.queue_on_disk.pop(0)
                    self.queue.append(cv2.imread(image_path))
                    os.remove(image_path)  # remove on disk
        return frame

    def has_next(self):
        return not self.queue == []

    def clear_all(self):
        self.queue.clear()
        self.clear_queue_on_disk()

    def clear_queue_on_disk(self):
        self.queue_on_disk.clear()
        if os.path.isdir(self.on_disk_save_path):
            shutil.rmtree(self.on_disk_save_path, ignore_errors=True)

    def can_write_to_queue(self):
        # if queue on disk is just full then it need to be empty before can write a gain
        if self.queue_on_disk_is_just_full:
            if self.queue_on_disk == []:
                self.queue_on_disk_is_just_full = False
                return True
            else:
                return False
        # else if it's normal
        else:
            if len(self.queue_on_disk) < self.max_frame_on_disk:
                return True
            # if queue_on_disk is full
            else:
                self.queue_on_disk_is_just_full = True
                logger.error(
                    'Number of images can be saved on disk is at limit')
                return False

    def stop(self):
        self.running = False

    def release(self):
        self.queue = []
        self.queue_on_disk = []
