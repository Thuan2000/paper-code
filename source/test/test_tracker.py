import unittest
from tracker import Tracker
from face_info import FaceInfo
import time
import numpy as np
import os


class TestTrackerMethod(unittest.TestCase):

    def setUp(self):
        self.tracker = Tracker(track_id=1)

    def mock_face_info(self):
        bounding_box = np.random.randint(100, size=(4))
        frame_id = 1
        face_image = np.ones((160,160,3), dtype=np.uint8)
        str_padded_bbox = '_'.join(bounding_box.astype('str').tolist())
        landmarks = np.random.randint(100, size=(10))
        return FaceInfo(bounding_box, frame_id, face_image,
                        str_padded_bbox, landmarks)

    
    def test_get_idle_time_should_return_nearly_exact_time(self):
        ''' asign a new time tick and check if get idle time return the right time '''
        start = time.time()
        self.tracker.tick = start
        print(self.tracker.tick, start)
        # because the time return will not exactly equal, so we need to round
        # to get a nearly equal
        time_span = time.time() - start
        idle_time = self.tracker.get_idle_time()
        self.assertAlmostEqual(time_span, idle_time, places=5)

    def test_get_track_time_return_nearly_exact_time(self):
        ''' asign a new start_time and check if get idle time return the right time '''
        start = time.time()
        self.tracker.start_time = start
        # because the time return will not exactly equal, so we need to round
        # to get a nearly equal
        time_span = time.time() - start
        idle_time = self.tracker.get_track_time()
        self.assertAlmostEqual(time_span, idle_time, places=5)

    def test_update_embeddings_should_assign_embs_into_elements(self):
        ''' assign new embs to elements and check if the exact element embs contain those input array '''
        # assing elements to tracker
        for i in range(10):
            self.tracker.elements.append(self.mock_face_info())
        nrof_elements = len(self.tracker.elements)
        start_id = np.random.randint(nrof_elements)
        interval = np.random.randint(nrof_elements - start_id)
        embeddings_array = np.random.random((interval, 128))
        self.tracker.update_embeddings(embeddings_array, start_id, interval)
        emb_array_idx = 0
        for element_idx in range(start_id, start_id + interval):
            self.assertTrue(np.array_equal(
                self.tracker.elements[element_idx].embedding,
                embeddings_array[emb_array_idx]))
            emb_array_idx += 1

    def test_random_samples_should_not_exceed_the_number_of_element_range(self):
        ''' test boundaries of return random elements sample '''
        nrof_elements_list = [0, 50, 100, 200]
        for nrof_elements in nrof_elements_list:
            max_nrof_elements = min(nrof_elements, 100)
            # init the elements
            self.tracker.elements = [1]*nrof_elements
            # do the random sample
            self.tracker.random_samples()
            self.assertEqual(len(self.tracker.elements), max_nrof_elements)

    def test_random_samples_return_elements_that_belong_to_current_tracker(self):
        ''' test if the return elements is from the original elements '''
        # do init
        input_elements_ids = []
        for i in range(200):
            new_element = self.mock_face_info()
            self.tracker.elements.append(new_element)
            input_elements_ids.append(id(new_element))
        # do checking
        self.tracker.random_samples()
        for element in self.tracker.elements:
            element_id = id(element)
            self.assertIn(element_id, input_elements_ids)

    def test_predict_should_change_the_tick_time(self):
        ''' mock the predict method of KCFTracker to test the tick tim change '''
        old_tick = self.tracker.tick
        self.tracker.tracker.predict = unittest.mock.MagicMock(return_value=True)
        self.tracker.predict('mock_frame')
        self.assertNotEqual(old_tick, self.tracker.tick)

        old_tick = self.tracker.tick
        self.tracker.tracker.predict = unittest.mock.MagicMock(return_value=False)
        self.tracker.predict('mock_frame')
        self.assertEqual(old_tick, self.tracker.tick)

    def test_update_detection_should_save_image_to_disk(self):
        ''' mock several face_info method to test the saving image process of this method '''
        self.tracker.tracker.start_track = unittest.mock.MagicMock(return_value=None)
        face_info = self.mock_face_info()
        face_info.is_good = unittest.mock.MagicMock(return_value=True)
        face_info.str_info = unittest.mock.MagicMock(return_value='sample')
        saved_img_path = os.path.join(self.tracker.track_id_path,
                                      '{}_{}.jpg'.format(self.tracker.track_id,
                                                         face_info.str_info()))
        print(saved_img_path)
        self.tracker.update_detection('mock_frame', face_info)
        # time.sleep(10)
        self.assertTrue(os.path.exists(saved_img_path))

    def test_generate_face_id_should_return_true_face_id_format(self):
        ''' random some input and check if the generated string is matched '''
        track_id = np.random.randint(10)
        start_time = time.time()
        area = 'VTV'
        expected_str = '{}-{}-{}'.format(area, track_id, start_time)
        self.tracker.track_id = track_id
        self.tracker.start_time = start_time

        self.assertEqual(expected_str, self.tracker.generate_face_id(area))

    def test_is_qualified_to_be_recognition_should_return_true(self):
        ''' mock some always true values to check method will return true '''
        self.tracker.get_track_time = unittest.mock.MagicMock(return_value=np.inf)
        self.tracker.elements = [1]*100000
        self.assertTrue(self.tracker.is_qualified_to_be_recognized())

    def branch_test_for_is_qualified_to_be_recognition_should_return_false(self):
        ''' mock differents inpuy value to test all the false cases of the method '''
        track_times = [0, 0, 0, 0, np.inf, np.inf, np.inf]
        elements_lens = [np.inf, np.inf, 0, 0, np.inf, 0, 0]
        assign_new_face_id = [False, True, False, True, True, False, True]

        for track_time, elements_len, do_assign in zip(track_times, elements_lens, assign_new_face_id):
            self.tracker.track_time = track_time
            self.tracker.elements = [1]*elements_len
            if do_assign:
                self.tracker.face_id = 'new'
            self.assertFalse(self.tracker.is_qualified_to_be_recognized())

    def test_deep_clone_should_return_the_same_attributes_value(self):
        self.tracker.elements = [self.mock_face_info(), self.mock_face_info()]
        cloned_tracker = self.tracker.deep_clone()
        self.assertEqual(cloned_tracker.label, self.tracker.label)
        self.assertEqual(cloned_tracker.is_tracking, self.tracker.is_tracking)
        self.assertEqual(cloned_tracker.tick, self.tracker.tick)
        self.assertEqual(cloned_tracker.start_time, self.tracker.start_time)
        self.assertEqual(cloned_tracker.track_id, self.tracker.track_id)
        self.assertEqual(cloned_tracker.face_id, self.tracker.face_id)
        self.assertEqual(cloned_tracker.represent_image_id, self.tracker.represent_image_id)
        self.assertEqual(cloned_tracker.send_time, self.tracker.send_time)
        self.assertEqual(cloned_tracker.is_new_face, self.tracker.is_new_face)
        self.assertEqual(cloned_tracker.track_id_path, self.tracker.track_id_path)

if __name__ == '__main__':
    unittest.main()
