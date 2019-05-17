from unittest.mock import patch
from onetime.generic_detection_tracking import generic_function
import imageio
import numpy as np
import os
import glob
from config import Config
import unittest


class FrameReader(object):

    def __init__(self, folder_path, scale_factor=1):
        self.frames = self.gen_frames(folder_path)
        self.frame_ids = sorted(list(self.frames.keys()))
        self.pointer = 0

    def next_frame(self):
        try:
            image = self.create_fake_frame(
                self.frames[self.frame_ids[self.pointer]])
            self.pointer += 1
            return image
        except IndexError:
            return None

    def get_info(self):
        '''
        Get information
        '''
        frame = None
        while frame is None:
            frame = self.next_frame()
        return 24, frame.shape[1], frame.shape[0]

    def create_fake_frame(self, detections):
        full_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for detection in detections:
            image = imageio.imread(detection[0])
            origin = detection[1]
            pad = detection[2]
            top_left_x = origin[0] - pad[0]
            top_left_y = origin[1] - pad[1]
            w = image.shape[1]
            h = image.shape[0]
            full_image[
                top_left_y - 16: top_left_y + h - 16,
                top_left_x - 16: top_left_x + w - 16, :] \
                = image
        return full_image

    def gen_frames(self, root_folder):
        dir_list = []
        for folder in os.listdir(root_folder):
            dir_list.append(folder)
        frames = {}
        for dir in dir_list:
            for file in glob.glob(os.path.join(root_folder, dir, '*')):
                file_name = os.path.split(file)[-1]
                frame_id = file_name.split('_')[5]
                origin_bbox = [int(i) for i in file_name.split('_')[1:5]]
                if file_name.startswith('BAD-TRACK'):
                    padding_bbox = [
                        int(i) for i in file_name.split('.')[0].split('_')[-4:]
                    ]
                else:
                    padding_bbox = [
                        int(i) for i in file_name.split('.')[1].split('_')[-4:]
                    ]
                if frame_id in frames:
                    frames[frame_id].append((file, origin_bbox, padding_bbox))
                else:
                    frames[frame_id] = [(file, origin_bbox, padding_bbox)]
        return frames

    def release(self):
        return None


class TestGeneric(unittest.TestCase):

    @patch('onetime.generic_detection_tracking.URLFrameReader')
    def test_generic_detection(self, frame_reader):
        frame_reader.return_value = FrameReader('../data/test_generic_data', 1)
        Config.Track.FACE_TRACK_IMAGES_OUT = True
        try:
            generic_function('../data/test_generic_data', None, 'VVT',
                             Config.FACENET_DIR, True)
        except KeyboardInterrupt:
            nrof_trackers = os.listdir('../data/tracking')
            self.assertEqual(2, len(nrof_trackers))
            self.assertEqual(75, len(os.listdir('../data/tracking/0')))
            self.assertEqual(7, len(os.listdir('../data/tracking/1')))


if __name__ == '__main__':
    unittest.main()
