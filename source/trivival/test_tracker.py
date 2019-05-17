import unittest
import numpy as np
from sklearn import neighbors
from tracker import TrackerResult, \
    TrackerResultsDict, \
    Tracker, \
    TrackersHistory, \
    TrackersList, \
    HistoryMatcher
from config import Config
from unittest.mock import patch
from unittest.mock import Mock, MagicMock
import cv2


class FakeMatcher():

    def build(self, embs, labels):
        self.labels = labels
        if type(embs) == np.ndarray:
            self.embs = embs
        else:
            self.embs = np.vstack(embs)
        self.tree = neighbors.KDTree(self.embs)

    def match(self,
              emb,
              top_matches=Config.Matcher.MAX_TOP_MATCHES,
              return_min_dist=False):
        dists, ids = self.tree.query(emb.reshape(1, -1), k=top_matches)
        dists = np.squeeze(dists)
        ids = np.squeeze(ids)
        convert_ids = [self.labels[id] for id in ids]
        dist = dists[0]
        if dist > 0.59:
            convert_ids[0] = 'NEW_FACE'
        return convert_ids[0], convert_ids, dist

    def query(self, embs_list, k):
        return self.tree.query(embs_list, k=k)

    def plus_one_numofids(self):
        pass


def create_tracker(elements):
    element_idx = 0
    tracker = None
    for element in elements:
        if element_idx == 0:
            tracker = Tracker(0, 0, [1, 1, 1, 1], element, element, 0, 0, 1, 0,
                              [1, 1, 1, 1])
        else:
            tracker.update_tracker(0, [1, 1, 1, 1], element, element, 0, 1, 0,
                                   [1, 1, 1, 1])
        element_idx += 1
    return tracker


class TestTrackerResult(unittest.TestCase):

    def test_init(self):
        tracker_result = TrackerResult()
        self.assertEqual(tracker_result.track_names, [])
        self.assertEqual(tracker_result.bounding_boxes, [])

    def test_append_result(self):
        tracker_result = TrackerResult()
        track_name = "test_track"
        track_bb = [0., 0., 0., 0.]
        for _ in range(10):
            tracker_result.append_result(track_name, track_bb)

        self.assertEqual(len(tracker_result.track_names), 10)
        self.assertEqual(len(tracker_result.bounding_boxes), 10)
        self.assertEqual(tracker_result.track_names[-1], track_name)
        self.assertEqual(tracker_result.bounding_boxes[-1], track_bb)

    def test_clear(self):
        tracker_result = TrackerResult()
        track_name = "test_track"
        track_bb = [0., 0., 0., 0.]
        for _ in range(10):
            tracker_result.append_result(track_name, track_bb)
        self.assertEqual(len(tracker_result.track_names), 10)
        self.assertEqual(len(tracker_result.bounding_boxes), 10)
        tracker_result.clear()
        self.assertEqual(tracker_result.track_names, [])
        self.assertEqual(tracker_result.bounding_boxes, [])


class TestTrackerResultsDict(unittest.TestCase):

    def test_init(self):
        result_dict = TrackerResultsDict()
        self.assertEqual(result_dict.tracker_results_dict, {})

    def test_update(self):
        result_dict = TrackerResultsDict()
        fid = 0
        track_name = "test_track"
        track_bb = [0., 0., 0., 0.]
        result_dict.update_dict(fid, track_name, track_bb)
        self.assertIn(fid, result_dict.tracker_results_dict)
        self.assertEqual(len(result_dict.tracker_results_dict), 1)
        result_dict.update_dict(fid, track_name, track_bb)
        self.assertEqual(len(result_dict.tracker_results_dict), 1)
        new_fid = 1
        result_dict.update_dict(new_fid, track_name, track_bb)
        self.assertEqual(len(result_dict.tracker_results_dict), 2)

    def test_update_two_dict(self):
        result_dict = TrackerResultsDict()
        self.assertIsNone(result_dict.update_two_dict({}), None)
        test_result_dict = TrackerResultsDict()
        fid = 0
        track_name = "test_track"
        track_bb = [0., 0., 0., 0.]
        test_result_dict.update_dict(fid, track_name, track_bb)
        result_dict.update_two_dict(test_result_dict)
        self.assertEqual(len(test_result_dict.tracker_results_dict), 1)
        self.assertIn(fid, result_dict.tracker_results_dict)


@patch("config.Config.Track.MIN_NOF_TRACKED_FACES", 30)
@patch("config.Config.Matcher.MAX_TOP_MATCHES", 3)
class TestTracker(unittest.TestCase):

    def test_init(self):
        track_id = 0
        time_stamp = 0
        bb = [0, 0, 0, 0]
        img = Mock()
        emb = np.arange(128)
        angle = 0
        area = 0
        frame_id = 0
        padded = [0, 0, 0, 0]
        tracker = Tracker(track_id, time_stamp, bb, img, emb, angle, area, 1,
                          frame_id, padded)
        self.assertEqual(len(tracker.elements), 1)

    def test_update_tracker(self):
        track_id = 0
        time_stamp = 0
        bb = [0, 0, 0, 0]
        img = Mock()
        emb = np.arange(128)
        angle = 0
        area = 0
        frame_id = 0
        padded = [0, 0, 0, 0]
        tracker = Tracker(track_id, time_stamp, bb, img, emb, angle, area, 1,
                          frame_id, padded)
        new_emb = np.array([1, 2, 3, 4])
        tracker.update_tracker(time_stamp, bb, img, new_emb, angle, 1, frame_id,
                               padded)
        self.assertTrue(np.array_equal(tracker.retrack_emb, new_emb))
        self.assertEqual(len(tracker.elements), 2)

    def test_tracker_recognizer(self):
        matcher = FakeMatcher()
        # prepare data
        embs = np.eye(12)
        labels = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        matcher.build(embs, labels)
        tracker = create_tracker([embs[i, :] for i in range(12)])
        self.assertEqual(12, len(tracker.elements))
        predict_face = tracker.tracker_recognizer(matcher)

        # test certain face id
        test_id_embs = [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])] * 50
        tracker = create_tracker(test_id_embs)
        predict_face = tracker.tracker_recognizer(matcher)
        self.assertEqual(50, len(tracker.elements))
        self.assertEqual(1, predict_face)

        test_id_embs1 = [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])] * 10
        test_id_embs2 = [np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])] * 10
        test_id_embs3 = [np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])] * 10
        test_id_embs4 = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])] * 10
        test_id_embs = test_id_embs1 + test_id_embs2 + test_id_embs3 + test_id_embs4
        tracker = create_tracker(test_id_embs)
        predict_face = tracker.tracker_recognizer(matcher)
        self.assertEqual('NEW_FACE', predict_face)


def fake_os(*fake_arg):
    return


class TestTrackerHistory(unittest.TestCase):

    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    def test_matcher_tracker(self):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        labels = ['TCH-VVT-' + str(i) for i in labels]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 20)
            tracker.name = 'TCH-VVT-' + str(i)
            trackers['TCH-VVT-' + str(i)] = tracker

        tracker_history = TrackersHistory()
        tracker_history.trackers = trackers
        tracker_history.current_id = 13
        tracker_history.start_time = 0
        tracker_history.labels = labels
        tracker_history.embs = [embs[i, :] for i in range(12)]
        history_matcher = HistoryMatcher()
        history_matcher.from_trackers_rebuild_matcher(trackers)
        tracker_history.history_matcher = history_matcher
        predicted_id = tracker_history.match_tracker(
            trackers['TCH-VVT-1'], match_mode="toolmatcher")
        self.assertEqual('TCH-VVT-1', predicted_id)

        # new face tracker
        vec = np.zeros(12)
        vec[1] = 1
        vec[8] = 1
        tracker = create_tracker([vec] * 30)
        pred_id = tracker_history.match_tracker(
            tracker, match_mode="toolmatcher")
        self.assertEqual('NEW_FACE', pred_id)

        # twist vector
        vec = np.zeros(12)
        vec[0] = 1.01
        tracker = create_tracker([vec] * 30)
        pred_id = tracker_history.match_tracker(
            tracker, match_mode="toolmatcher")
        self.assertEqual('TCH-VVT-0', pred_id)

    @patch("os.mkdir")
    def test_bad_track_add_tracker(self, mock_mkdir):
        mock_mkdir.side_effect = fake_os
        vec = np.zeros(12)
        vec[0] = 1.01
        tracker = create_tracker([vec] * 30)
        tracker.name = 'BAD-TRACK'
        tracker_history = TrackersHistory()
        result = tracker_history.add_tracker(tracker, Mock(), Mock())
        a, b = result
        self.assertTrue(type(a) == TrackerResultsDict)
        self.assertTrue(type(b) == dict)

    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    @patch("config.Config.Track.MIN_MATCH_DISTACE_OUT", True)
    @patch("config.Config.Track.FACE_TRACK_IMAGES_OUT", True)
    @patch("config.Config.Track.PREDICT_DICT_OUT", True)
    @patch("config.Config.SEND_QUEUE_TO_DASHBOARD", True)
    @patch("os.mkdir")
    @patch("cv2.cvtColor", return_value=True)
    @patch("cv2.imwrite", return_value=True)
    def test_add_tracker(self, mock_write, mock_color, mock_mkdir):
        mock_mkdir.side_effect = fake_os
        print("\ntest add tracker")
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        tracker_history = TrackersHistory()
        tracker_history.trackers = trackers
        tracker_history.current_id = 13
        tracker_history.start_time = 0
        tracker_history.labels = labels
        tracker_history.embs = [embs[i, :] for i in range(12)]
        history_matcher = neighbors.KDTree(
            embs, leaf_size=Config.Matcher.INDEX_LEAF_SIZE, metric='euclidean')
        tracker_history.history_matcher = history_matcher
        mock = MagicMock(side_effect=fake_os)
        with patch('os.mkdir', mock):
            tracker_history.add_tracker(trackers[0], matcher, Mock())

    # @patch("config.Config.Matcher.EMB_LENGTH", 12)
    # def test_update_kdtree(self):
    #     matcher = FakeMatcher()
    #     embs = np.eye(12)
    #     labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #     matcher.build(embs, labels)
    #     # total 12 tracker
    #     zero = np.zeros(12)
    #     trackers = {}
    #     for i in range(12):
    #         vec = zero.copy()
    #         vec[i] = 1
    #
    #         tracker = create_tracker([vec] * 40)
    #         trackers[i] = tracker
    #     trackers_history = TrackersHistory()
    #     trackers_history.trackers = trackers
    #     trackers_history.self_update_kdtree()

    def test_confirm_id(self):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        token = 'TCH-VoVanTan-118-1522571895.8298123_401_183_447_242_87075'
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            tracker.track_id = 118
            trackers[i] = tracker
        trackers_history = TrackersHistory()
        trackers_history.trackers = trackers
        trackers_history.confirm_id({
            token: 'TCH-VoVanTan-118-1522571895.8298123'
        })
        trackers_history.confirm_id([])

    @patch("utils.PickleUtils.read_pickle", return_value={})
    @patch("utils.PickleUtils.save_pickle", return_value=True)
    def test_extract_register(self, mock_save_pickle, mock_read_pickle):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            tracker.track_id = 118
            trackers[i] = tracker
        trackers_history = TrackersHistory()
        trackers_history.trackers = trackers
        trackers_history.extract_register()

    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    def test_clear_history(self):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker@patch("config.Config.Matcher.EMB_LENGTH", 12)
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        trackers_history = TrackersHistory()
        trackers_history.trackers = trackers
        trackers_history.clear_history([0, 1, 2])
        result = trackers_history.clear_history([3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(-1, result)
        self.assertEqual(None, trackers_history.history_matcher.matcher)
        result = trackers_history.clear_history([])
        self.assertEqual(-1, result)


class TestTrackerList(unittest.TestCase):

    def test_retrack_tracker_cannot_retrack(self):
        tracker_list = TrackersList()
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        tracker_list.trackers = trackers
        result = tracker_list.get_retrack_tracker(Mock(), 0)
        self.assertEqual((-1, -1), result)

    def test_retrack_tracker(self):
        tracker_list = TrackersList()
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        tracker_list.trackers = trackers
        result = tracker_list.get_retrack_tracker(
            embs[0, :], 1, retrack_mode='minrate')
        self.assertEqual((1.0, 0), result)

    @patch("config.Config.Track.RETRACK_MINRATE", 1.0)
    def test_retrack_tracker_change_minrate(self):
        tracker_list = TrackersList()
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        tracker_list.trackers = trackers
        result = tracker_list.get_retrack_tracker(
            embs[0, :], 1, retrack_mode='minrate')
        self.assertEqual((-1, -1), result)

    def test_update_trackers_list(self):
        tracker_list = TrackersList()
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        tracker_list.trackers = trackers
        current = tracker_list.current_track_id
        tracker_list.update_trackers_list(14, 0, [0, 0, 0, 0], 'image', Mock(),
                                          0, 0, 1, 0, [0, 0, 0, 0], matcher,
                                          Mock())
        self.assertEqual(current + 1, tracker_list.current_track_id)

        tracker_list.update_trackers_list(1, 0, [0, 0, 0, 0], 'image', Mock(),
                                          0, 0, 1, 0, [0, 0, 0, 0], matcher,
                                          Mock())

    def test_match_face_with_new_tracker(self):
        frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(frame, (0, 0), (20, 20), (0, 255, 0), 3)
        # tracker = dlib.correlation_tracker()
        # tracker.start_track(frame, dlib.rectangle(0, 0, 20, 20))
        # dlib_tracker = (tra)
        tracker_list = TrackersList()
        fid = tracker_list.matching_face_with_trackers(frame, 0, [0, 0, 20, 20],
                                                       Mock(), 1)
        self.assertEqual(1, len(tracker_list.dlib_trackers))
        self.assertEqual(0, fid)

    def test_match_face_with_new_tracker_no_use(self):
        frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(frame, (0, 0), (20, 20), (0, 255, 0), 3)
        tracker_list = TrackersList()
        tracker_list.matching_face_with_trackers(frame, 0, [0, 0, 20, 20],
                                                 Mock(), 1)
        new_frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(new_frame, (5, 5), (25, 25), (0, 255, 0), 3)
        tracker_list.matching_face_with_trackers(new_frame, 0, [5, 5, 25, 25],
                                                 Mock(), 1)
        tracker_list.matching_face_with_trackers(new_frame, 1, [5, 5, 25, 25],
                                                 Mock(), 1)

    @patch("utils.check_overlap")
    def test_match_face_with_new_tracker_large_iou(self, mock_overlap):
        mock_overlap.return_value = 1.0
        frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(frame, (0, 0), (20, 20), (0, 255, 0), 3)
        tracker_list = TrackersList()
        tracker_list.matching_face_with_trackers(frame, 0, [0, 0, 20, 20],
                                                 Mock(), 1)
        new_frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(new_frame, (5, 5), (25, 25), (0, 255, 0), 3)
        tracker_list.matching_face_with_trackers(new_frame, 0, [5, 5, 25, 25],
                                                 Mock(), 1)

    @patch("utils.check_overlap")
    @patch("tracker.TrackersList.get_retrack_tracker")
    def test_match_face_dlib_but_not_overlap(self, retrack, mock_overlap):
        mock_overlap.return_value = 1.0
        retrack.return_value = 1.0, 1
        tracker_list = TrackersList()
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1

            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker

        tracker_list.trackers = trackers

        frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(frame, (0, 0), (20, 20), (0, 255, 0), 3)
        tracker_list.matching_face_with_trackers(frame, 0, [0, 0, 20, 20],
                                                 Mock(), 1)
        new_frame = np.zeros((512, 512, 3), np.uint8)
        cv2.rectangle(new_frame, (5, 5), (25, 25), (0, 255, 0), 3)
        tracker_list.matching_face_with_trackers(new_frame, 0, [5, 5, 25, 25],
                                                 embs[0, :], 1)

    def test_check_recognize_tracker_detecting_name(self):
        '''
        What?

        input:
        matcher
        track_id
        rabbit_mq
        :return:
        '''
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 40)
            tracker.name = 'NOT_DETECTING'
            trackers[i] = tracker
        tracker_list = TrackersList()
        tracker_list.trackers = trackers
        result = tracker_list.check_recognize_tracker(Mock(), Mock(), 0)
        self.assertEqual(-1, result)

    def test_check_recognize_tracker_detecting_name_bad_track(self):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 40)
            trackers[i] = tracker
        tracker_list = TrackersList()
        tracker_list.trackers = trackers
        result = tracker_list.check_recognize_tracker(Mock(), Mock(), 1)
        self.assertEqual(-1, result)
        self.assertEqual('BAD-TRACK', tracker_list.trackers[1].name)

    @patch("config.Config.Track.CURRENT_EXTRACR_TIMER", float('inf'))
    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    def test_check_recognize_tracker_detecting_name_less_element(self):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 2)
            trackers[i] = tracker
        tracker_list = TrackersList()
        tracker_list.trackers = trackers
        result = tracker_list.check_recognize_tracker(matcher, Mock(), 1)
        self.assertTrue(result)

    @patch("config.Config.Track.CURRENT_EXTRACR_TIMER", float('inf'))
    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    @patch("scipy.misc.imsave")
    def test_check_recognize_tracker_mergeable(self, mock_save):

        def fake_img_save(arg1, arg2):
            return True

        mock_save.side_effect = fake_img_save
        matcher = HistoryMatcher()
        # embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 30)
            tracker.name = 'TCH-VVT-' + str(i)
            trackers['TCH-VVT-' + str(i)] = tracker

        matcher.from_trackers_rebuild_matcher(trackers)

        vec = np.zeros(12)
        vec[1] = 1
        new_tracker = create_tracker([vec] * 30)

        tracker_list = TrackersList()
        tracker_list.trackers = {1: new_tracker}
        tracker_list.current_trackers_history.trackers = trackers
        tracker_list.current_trackers_history.history_matcher = matcher
        tracker_list.current_trackers_history.labels = labels
        result = tracker_list.check_recognize_tracker(matcher, Mock(), 1)
        self.assertTrue(result)
        self.assertEqual('TCH-VVT-1', tracker_list.trackers[1].name)

    @patch("config.Config.Track.CURRENT_EXTRACR_TIMER", float('inf'))
    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    @patch("scipy.misc.imsave")
    def test_check_recognize_tracker_not_mergeable_new_face(self, mock_save):

        def fake_img_save(arg1, arg2):
            return True

        mock_save.side_effect = fake_img_save
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 6)
            trackers[i] = tracker

        vec = np.zeros(12)
        vec[1] = 1
        vec[2] = 1
        vec[3] = 2
        vec[4] = 3
        new_tracker = create_tracker([vec] * 6)

        tracker_list = TrackersList()
        tracker_list.trackers = {1: new_tracker}
        tracker_list.current_trackers_history.trackers = trackers
        tracker_list.current_trackers_history.history_matcher = matcher
        tracker_list.current_trackers_history.labels = labels
        tracker_list.check_recognize_tracker(matcher, Mock(), 1)

    @patch("config.Config.Track.CURRENT_EXTRACR_TIMER", float('inf'))
    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    @patch("scipy.misc.imsave")
    @patch("config.Config.Track.SEND_FIRST_STEP_RECOG_API", True)
    @patch("config.Config.Track.USE_FIRST_STEP_RECOG_DASHBOARD", True)
    def test_check_recognize_tracker_not_mergeable_old_face(self, mock_save):

        def fake_img_save(arg1, arg2):
            return True

        mock_save.side_effect = fake_img_save
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 30)
            tracker.name = 'TCH-VVT-' + str(i)
            trackers[i] = tracker

        vec = np.zeros(12)
        vec[1] = 1
        vec[2] = 1
        vec[3] = 2
        vec[4] = 3
        new_tracker = create_tracker([vec] * 30)

        history_matcher = HistoryMatcher()
        history_matcher.from_trackers_rebuild_matcher(trackers)

        tracker_list = TrackersList()
        tracker_list.trackers = {1: new_tracker}
        tracker_list.current_trackers_history.trackers = trackers
        tracker_list.current_trackers_history.history_matcher = history_matcher
        tracker_list.current_trackers_history.labels = labels
        recognizer_matcher = FakeMatcher()
        recog_embs = np.vstack((vec for _ in range(12)))
        recognizer_matcher.build(recog_embs, ['13'] * 12)
        tracker_list.check_recognize_tracker(recognizer_matcher, Mock(), 1)

    def test_delete_tracker_history_false(self):
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 6)
            trackers[i] = tracker

        vec = np.zeros(12)
        vec[1] = 1
        vec[2] = 1
        vec[3] = 2
        vec[4] = 3
        new_tracker = create_tracker([vec] * 6)
        tracker_list = TrackersList()
        tracker_list.trackers = {1: new_tracker}
        tracker_list.check_delete_trackers(matcher, Mock(), history_mode=False)

    @patch("os.mkdir")
    @patch("config.Config.Matcher.EMB_LENGTH", 12)
    def test_delete_tracker_history_True(self, mock_dir):
        mock_dir.side_effect = fake_os('123')
        matcher = FakeMatcher()
        embs = np.eye(12)
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        matcher.build(embs, labels)
        # total 12 tracker
        zero = np.zeros(12)
        trackers = {}
        for i in range(12):
            vec = zero.copy()
            vec[i] = 1
            tracker = create_tracker([vec] * 6)
            tracker.area = "0"
            trackers[i] = tracker

        vec = np.zeros(12)
        vec[1] = 1
        vec[2] = 1
        vec[3] = 2
        vec[4] = 3
        new_tracker = create_tracker([vec] * 6)
        new_tracker.area = '0'
        tracker_list = TrackersList()
        tracker_list.trackers = {1: new_tracker}
        tracker_list.check_delete_trackers(matcher, Mock(), history_mode=True)


if __name__ == '__main__':
    unittest.main()
