import unittest
import time
import threading
import numpy as np
from unittest import mock
from frame_queue import FrameQueue
from frame_reader import URLFrameReader


class TestFrameQueue(unittest.TestCase):

    def setUp(self):
        frame_reader = URLFrameReader('')
        self.frame_shape = (480, 720, 3)
        sample_frame = np.zeros(self.frame_shape)
        frame_reader.next_frame = mock.MagicMock(return_value=sample_frame)
        self.lock = threading.Lock()
        self.max_queue_size = 10
        self.max_frame_on_disk = 10
        self.frame_queue = FrameQueue(
            frame_reader,
            self.lock,
            max_queue_size=self.max_queue_size,
            max_frame_on_disk=self.max_frame_on_disk)
        self.frame_queue.start()

    def tearDown(self):
        self.frame_queue.stop()

    def test_max_frame_in_queues(self):
        self.assertLessEqual(len(self.frame_queue.queue), self.max_queue_size)
        self.assertLessEqual(
            len(self.frame_queue.queue_on_disk), self.max_frame_on_disk)

    def test_frame_queue_priority(self):
        if len(self.frame_queue.queue) < self.max_queue_size:
            self.assertEqual(len(self.frame_queue.queue_on_disk), 0)

    def test_has_next_frame(self):
        self.assertEqual(self.frame_queue.has_next(),
                         len(self.frame_queue.queue) > 0)

    def test_get_next_frame(self):
        if self.frame_queue.has_next():
            self.assertEqual(self.frame_queue.next_frame().shape,
                             self.frame_shape)
        else:
            self.assertIsNone(self.frame_queue.next_frame())

    def test_clear_all(self):
        self.frame_queue.stop()
        self.frame_queue.clear_all()
        self.assertLessEqual(len(self.frame_queue.queue), 0)
        self.assertLessEqual(len(self.frame_queue.queue_on_disk), 0)


if __name__ == '__main__':
    unittest.main()
