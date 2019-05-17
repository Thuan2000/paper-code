import unittest
from unittest import mock
import numpy as np
from tracker_manager import TrackerManager
from tracker import Tracker
from face_info import FaceInfo


class TestTrackerManager(unittest.TestCase):

    def setUp(self):
        self.tracker_manager = TrackerManager('test')

    def test_checking_early_recognition_should_return_tracker_if_qualified(self):
        ''' mock is_qualified_to_be_recognized() always return True to test if tracker_manager do return '''
        Tracker.is_qualified_to_be_recognized = mock.MagicMock(return_value=True)
        current_trackers = []
        for i in range(10):
            new_tracker = Tracker(track_id=i)
            self.tracker_manager.current_trackers[i] = new_tracker
            current_trackers.append(new_tracker)
        
        qualified_trackers = self.tracker_manager.checking_early_recognition()
        for q_tracker, c_tracker in zip(qualified_trackers, current_trackers):
            self.assertEqual(id(q_tracker), id(c_tracker))

    def test_cleanup_overtime_tracker_should_return_tracker_if_exceed_exists_time(self):
        ''' mock timeout=0 to test cleannup always return every trackers '''
        current_trackers = []
        for i in range(10):
            new_tracker = Tracker(track_id=i)
            self.tracker_manager.current_trackers[i] = new_tracker
            current_trackers.append(new_tracker)
        
        overtime_trackers = self.tracker_manager.cleanup_overtime_tracker(timeout=0)
        for q_tracker, c_tracker in zip(overtime_trackers, current_trackers):
            self.assertEqual(q_tracker.track_id, c_tracker.track_id)

    def test_associate_detections_to_trackers_should_complete_the_linear_assignment(self):
        ''' this method is used to assign new bboxes to current trackers base on iou '''
        trackers_bboxes = [
            [100, 100, 200, 200],
            [200, 200, 300, 300],
            [300, 300, 400, 400],
            [400, 400, 500, 500],
            [500, 500, 600, 600]
        ]
        detection_bboxes = [
            [100, 100, 200, 200],
            [300, 300, 400, 400],
            [600, 600, 700, 700]
        ]
        matches, unmatch_detections = \
            self.tracker_manager.associate_detections_to_trackers(
                trackers_bboxes, detection_bboxes)

        matches_groundtruth = np.array([[[0,0]], [[1,2]]])
        self.assertTrue(np.array_equal(matches, matches_groundtruth))
        self.assertEqual(unmatch_detections, [2])

    def mock_face_info(self):
        bounding_box = np.random.randint(100, size=(4))
        frame_id = 1
        face_image = np.ones((160,160,3), dtype=np.uint8)
        str_padded_bbox = '_'.join(bounding_box.astype('str').tolist())
        landmarks = np.random.randint(100, size=(10))
        return FaceInfo(bounding_box, frame_id, face_image,
                        str_padded_bbox, landmarks)

    def test_update_iou_tracking_should_update_new_sucessed_assign_tracker(self):
        ''' mock faceInfos data to check if this method do update_detection and create new tracker '''
        # setup tracker_manager and face_infos
        face_infos = [self.mock_face_info(), self.mock_face_info()]
        for i in range(3):
            new_tracker = Tracker(track_id=i)
            new_tracker.is_tracking = True
            self.tracker_manager.current_trackers[i] = new_tracker
            self.tracker_manager.id_counter += 1

        frame = np.ones((480,480,3), dtype=np.uint8)
        matches = np.array([[[0,0]]])
        unmatch_detections = [1]

        # mock methods
        Tracker.update_detection = mock.MagicMock(return_value=True)
        Tracker.get_pos = mock.MagicMock(return_value=True)
        self.tracker_manager.associate_detections_to_trackers = \
            mock.MagicMock(return_value=(matches, unmatch_detections))
        self.tracker_manager.update_iou_tracking(frame, face_infos)

        self.assertTrue(self.tracker_manager.id_counter-1 in \
            self.tracker_manager.current_trackers)
        self.assertEqual(len(self.tracker_manager.current_trackers), 4)


if __name__ == '__main__':
    unittest.main()