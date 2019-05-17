from config import Config
from cv_utils import PickleUtils
import unittest
import os
from matcher import KdTreeMatcher, FaissMatcher
import time
from collections import defaultdict
from database import DashboardDatabase


class MatcherTest(unittest.TestCase):
    '''
    Test matcher with small dataset
    '''

    def setUp(self):
        self.start_time = time.time()
        self.matcher = MatcherToTest()
        self.database = DashboardDatabase()



    def tearDown(self):
        t = time.time() - self.start_time
        print("%s: %.3f" % (self.id(), t))

    def test_matcher_on_init(self):
        '''
        Test newly created matcher, check if return NEW_FACE
        '''
        for face_id, embs in UNKNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids = self.matcher.match(emb, 2)
                self.assertEqual(['NEW_FACE'], top_ids)

    def test_build_matcher_from_reg_dict(self):
        '''
        Test build global matcher from face dict
        '''
        self.matcher.build(self.database)
        embs = []
        labels = []
        for face_id, face_embs in KNOW_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids
        self.matcher.fit(embs, labels)

        for face_id, embs in KNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids, min_dist = self.matcher.match(emb, 1, return_dists=True)
                self.assertEqual(top_ids[0], face_id)
                self.assertAlmostEqual(min_dist[0], 0, delta=1e-7)

    def test_build_matcher_from_reg_dict_using_image_id(self):
        '''
        Test build global matcher from face dict
        '''
        self.matcher.build(self.database)
        embs = []
        labels = []
        for face_id, face_embs in KNOW_IMGID_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids
        self.matcher.fit(embs, labels)

        for image_id, embs in KNOW_IMGID_EMBS_DICT.items():
            for emb in embs:
                top_ids, min_dist = self.matcher.match(emb, 1, return_dists=True)
                self.assertEqual(top_ids[0], image_id)
                self.assertAlmostEqual(min_dist[0], 0, delta=1e-7)

    def test_save_and_load_matcher(self):
        '''
        Loaded model must perform the same as saved model
        '''
        # build matcher and save
        self.matcher.build(self.database)
        self.matcher.save_model("temp.pkl")

        # load model to new matcher, make sure both matcher match
        new_matcher = MatcherToTest()
        new_matcher.load_model("temp.pkl")
        for face_id, embs in KNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids1, dist1 = self.matcher.match(emb, 2, return_dists=True)
                top_ids2, dist2 = new_matcher.match(emb, 2, return_dists=True)
                self.assertEqual(top_ids1[0], top_ids2[0])
                self.assertAlmostEqual(dist1[0], dist2[0], delta=1e-6)

    def test_matcher_fit(self):
        '''
        Test matcher fit match the fited label
        '''
        # build matcher on KNOWN set
        embs = []
        labels = []
        for face_id, face_embs in KNOW_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids
        self.matcher.fit(embs, labels)

        # test if KNOWN set match
        for face_id, embs in KNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids, dist = self.matcher.match(emb, 2, return_dists=True)
                self.assertEqual(face_id, top_ids[0])
                self.assertAlmostEqual(0, dist[0], delta=1e-7)

    def test_matcher_update(self):
        '''
        Test matcher update add to existing matcher
        '''
        # build matcher from KNOWN set
        embs = []
        labels = []
        for face_id, face_embs in KNOW_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids
        self.matcher.fit(embs, labels)

        # test that UNKNOWN set face has dist > 0
        for face_id, embs, in UNKNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids, dist = self.matcher.match(emb, 2, return_dists=True)
                self.assertNotAlmostEqual(dist[0], 0)

        # update matcher using UNKNOWN set
        embs = []
        labels = []
        for face_id, face_embs in UNKNOW_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids

        self.matcher.update(embs, labels)

        # test that dist in KNOWN set = 0
        for face_id, embs in KNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids, dist = self.matcher.match(emb, 2, return_dists=True)
                self.assertEqual(face_id, top_ids[0])
                self.assertAlmostEqual(0, dist[0], delta=1e-7)

        # test that dist in UNKNOWN set now also = 0
        for face_id, embs in UNKNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids, dist = self.matcher.match(emb, 2, return_dists=True)
                self.assertEqual(face_id, top_ids[0])
                self.assertAlmostEqual(0, dist[0], delta=1e-7)

    def test_fit_rebuild_new_matcher(self):
        '''
        Test if fit return new matcher
        '''
        # first fit on KNOWN set
        embs = []
        labels = []
        for face_id, face_embs in KNOW_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids
        self.matcher.fit(embs, labels)

        # rebuild matcher by fiting on UNKNOWN set
        embs = []
        labels = []
        for face_ids, face_embs in UNKNOW_EMBS_DICT.items():
            face_ids = [face_id] * len(face_embs)
            embs = embs + face_embs
            labels = labels + face_ids
        self.matcher.fit(embs, labels)

        # check that KNOWN set dist > 0
        for face_id, embs in KNOW_EMBS_DICT.items():
            for emb in embs:
                top_ids, dist = self.matcher.match(emb, 2, return_dists=True)
                self.assertNotAlmostEqual(dist[0], 0)


if __name__ == '__main__':
    reg_face_dict = PickleUtils.read_pickle(
        Config.PickleFile.REG_IMAGE_FACE_DICT_FILE, default={})
    face_ids = list(set(reg_face_dict.values()))
    known_faces = face_ids[:-5]
    unknown_faces = face_ids[-5:]
    print('Know face', known_faces)
    print('Unknow face', unknown_faces)
    KNOW_EMBS_DICT = defaultdict(list)
    KNOW_IMGID_EMBS_DICT = defaultdict(list)
    UNKNOW_EMBS_DICT = defaultdict(list)
    for image_id, face_id in reg_face_dict.items():
        emb = PickleUtils.read_pickle(
            os.path.join('/mnt/data/tch-data/tch-data/session/live/', '{}.pkl'.format(image_id)))[1]
        if face_id in known_faces:
            KNOW_EMBS_DICT[face_id].append(emb)
            KNOW_IMGID_EMBS_DICT[image_id].append(emb)
        else:
            UNKNOW_EMBS_DICT[face_id].append(emb)

    matcher = 'FA'
    if matcher == 'KD':
        MatcherToTest = KdTreeMatcher
        TUPLE_FILE = Config.PickleFile.MATCHER_TUP_FILE
    elif matcher == 'FA':
        MatcherToTest = FaissMatcher
        TUPLE_FILE = Config.PickleFile.FAISS_MATCHER_TUP_FILE

    print('Testing', MatcherToTest.__name__)

    unittest.main()
