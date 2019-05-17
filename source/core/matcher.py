from abc import ABCMeta, abstractmethod
import os
import faiss
import hnswlib
import time
import sys
from collections import namedtuple
import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from core.cv_utils import PickleUtils
from sklearn.svm import SVC
from utils import rwlock
from config import Config

KdTreeTuple = namedtuple('KdTreeTuple', ['embs_arr', 'labels_arr', 'length'])
FaissTuple = namedtuple('FaissTuple', ['embs_arr', 'labels_arr', 'length'])
SvmTuple = namedtuple('SvmTuple', ['embs_arr', 'labels_arr', 'length'])


class AbstractMatcher(metaclass=ABCMeta):

    def __init__(self, threshold=Config.Matcher.MIN_ASSIGN_THRESHOLD):
        '''
        :param threhold: matching threshold for this matcher
        :param indexing_time: how often this matcher will update itself
        '''
        self._threshold = threshold
        self._matcher_tup = None
        self._classifier = None
        self._rwlock = rwlock.RWLock()
        print("Abstract matcher")

    def load_model(self, file_name):
        # load matcher from pkl file
        self._matcher_tup = PickleUtils.read_pickle(file_name, default=None)
        if self._matcher_tup is not None:
            self._fit_array(self._matcher_tup.embs_arr,
                            self._matcher_tup.labels_arr)

    def save_model(self, file_name):
        # save matcher to pkl file
        PickleUtils.save_pickle(file_name, value=self._matcher_tup)

    def fit(self, embs_ls, labels_ls):
        '''
        # TODO: This is a temporal interface, as old fit version
        accept python list as input, not numpy array
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        pass
        self._fit_array(np.array(embs_ls), np.array(labels_ls))

    @abstractmethod
    def _fit_array(self, embs_arr, labels_arr):
        '''
        Fit current matcher to new embs and labels
        :param embs_arr: shape (n, Config.Matcher.EMB_LENGTH)
        :param labels_arr: shape (n)
        '''
        pass

    def update(self, new_embs_ls, new_labels_ls):
        '''
        # TODO: This is a temporal interface, as old fit version
        accept python list as input, not numpy array
        Fit current matcher to new embs and labels
        add new embs and labels to current matcher
        :param new_embs: list of embs
        :param new_labels: list of label (face id) for each emb
        '''
        self._update_array(np.array(new_embs_ls), np.array(new_labels_ls))

    @abstractmethod
    def _update_array(self, new_embs_arr, new_labels_arr):
        pass

    def build(self, database, imageid_to_keyid=None, use_image_id=False):
        '''
        build matcher using register file
        :param register_file: {image_id: face_id}
        '''
        # TODO: remore or keep use_image_id
        # TODO: move imageid_to_keyid outside this
        labels, embs = database.get_labels_and_embs()
        if labels is not None:
            self._fit_array(embs, labels)

    @abstractmethod
    def match(self, emb, top_matches, return_dists=False):
        # TODO: This function accept 1 emb of (Config.Matcher.EMB_LENGTH,), refactor to use match_arr instead
        '''
        Find the neart id the the embedding
        :param emb: Config.Matcher.EMB_LENGTHx1 embedding vector
        :param top_matches: number of closest match to emb
        :param min_dist: should return min distance
        :return track_id: closest id to emb
        :return top_match_ids: X top matches face
        :return dist
        '''
        pass


class KdTreeMatcher(AbstractMatcher):
    '''
    Find neareast id for a embedding by using kd-tree
    '''

    def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES,\
              threshold=None, return_dists=False):
        '''
        See superclass doc
        '''
        emb = emb.reshape((-1, Config.Matcher.EMB_LENGTH))
        match = self.match_arr(emb, top_matches, threshold, True)
        top_match_ids, dists = match
        if return_dists:
            return top_match_ids[0], dists[0]
        else:
            return top_match_ids[0]

    # TODO: merge two function
    def match_arr(self, embs, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold=None, return_dists=False):
        # embs: (n, Config.Matcher.EMB_LENGTH)
        if threshold is None:
            threshold = self._threshold

        if self._classifier is not None:
            self._rwlock.reader_acquire()
            top_matches = min(top_matches,
                              self._matcher_tup.embs_arr.shape[0] - 1)
            dists, inds = self._classifier.query(
                embs, k=top_matches)  # (n, top_matches)
            predict_ids = self._matcher_tup.labels_arr[inds[:, 0]]  # (n)
            min_dist = dists[:, 0]
            predict_ids[min_dist > self._threshold] = Config.Matcher.NEW_FACE
            top_match_ids = self._matcher_tup.labels_arr[inds]
            self._rwlock.reader_release()
        else:
            top_match_ids = np.full((embs.shape[0], 1), Config.Matcher.NEW_FACE)
            # predict_ids = np.full(embs.shape[0], Config.Matcher.NEW_FACE)
            dists = np.full((embs.shape[0], 1), -1)

        if return_dists:
            return top_match_ids.tolist(), dists.tolist()
        return top_match_ids.tolist()
        #TODO: Merge david logic
        '''
            predict_id = self._matcher_tup.labels[inds[0]]
            min_dist = dists[0]
            if min_dist <= threshold:
                top_match_ids = [self._matcher_tup.labels[idx] for (i, idx) in enumerate(inds) if dists[i] <= threshold]
            else:
                top_match_ids = [predict_id]
            dists = dists[0:len(top_match_ids)]
        '''

    def _fit_array(self, embs_arr, labels_arr):
        '''
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        length = embs_arr.shape[0]
        if length > 0:
            self._rwlock.writer_acquire()
            # reg_mat = np.asarray(embs).reshape((length, Config.Matcher.EMB_LENGTH))
            self._classifier = neighbors.KDTree(
                embs_arr,
                leaf_size=Config.Matcher.INDEX_LEAF_SIZE,
                metric='euclidean')
            self._matcher_tup = KdTreeTuple(embs_arr, labels_arr, length)
            self._rwlock.writer_release()
        else:
            self._matcher_tup = None
            self._classifier = None

    def _update_array(self, new_embs_arr, new_labels_arr):
        '''
        add new embs and labels to current matcher
        :param new_embs: list of embs
        :param new_labels: list of label (face id) for each emb
        '''
        if self._matcher_tup is None:
            self._fit_array(new_embs_arr, new_labels_arr)
        else:
            self._rwlock.writer_acquire()
            old_embs_arr = self._matcher_tup.embs_arr
            old_labels_arr = self._matcher_tup.labels_arr
            embs_arr = np.vstack((old_embs_arr, new_embs_arr))
            labels_arr = np.concatenate((old_labels_arr, new_labels_arr))
            self._rwlock.writer_release()

            # yet another rwlock
            self._fit_array(embs_arr, labels_arr)


class FaissMatcher(AbstractMatcher):
    '''
    Classify face id using Faiss
    '''

    def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold=None, return_dists=False):
        emb = emb.reshape((-1, Config.Matcher.EMB_LENGTH))
        match = self.match_arr(emb, top_matches, threshold, True)
        top_match_ids, dists = match
        if return_dists:
            return top_match_ids[0], dists[0]
        else:
            return top_match_ids[0]

    def match_arr(self, embs, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold=None, return_dists=False):
        # embs (n, Config.Matcher.EMB_LENGTH)
        if threshold is None:
            threshold = self._threshold
        if self._classifier is not None:
            top_matches = min(top_matches, self._matcher_tup.length - 1)
            self._rwlock.reader_acquire()
            dists, inds = self._classifier.search(
                embs.astype('float32'), k=top_matches)
            predict_ids = self._matcher_tup.labels_arr[inds[:, 0]]
            min_dist = dists[:, 0]
            predict_ids[min_dist > self._threshold] = Config.Matcher.NEW_FACE
            top_match_ids = self._matcher_tup.labels_arr[inds]
            self._rwlock.reader_release()
        else:
            top_match_ids = np.full((embs.shape[0], 1), Config.Matcher.NEW_FACE)
            # predict_ids = np.full(embs.shape[0], Config.Matcher.NEW_FACE)
            dists = np.full((embs.shape[0], 1), -1)

        if return_dists:
            return top_match_ids.tolist(), dists.tolist()
        return top_match_ids.tolist()
        # TODO: Merge david logic
        '''
            if min_dist <= threshold:
                top_match_ids = [self._matcher_tup.labels[idx] for (i, idx) in enumerate(inds) if dists[i] <= threshold]
            else:
                if always_return_closest:
                    top_match_ids = [predict_id]
                else:
                    top_match_ids = []
            dists = dists[0:len(top_match_ids)]
        '''

    def _fit_array(self, embs_arr, labels_arr):
        '''
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        length = embs_arr.shape[0]
        if length > 0:
            # only fit if we have data
            self._rwlock.writer_acquire()
            cpu_classifier = faiss.IndexFlatL2(Config.Matcher.EMB_LENGTH)

            # This line allow faiss to run on GPU, there is another function
            # that allow a certainly number of GPUs, e.g: 4
            # but we use this function in this code because our GPUtil limited
            # number of GPU used
            self._classifier = faiss.index_cpu_to_all_gpus(cpu_classifier)
            self._classifier.add(embs_arr.astype('float32'))
            self._matcher_tup = FaissTuple(embs_arr, labels_arr, length)
            self._rwlock.writer_release()
        else:
            self._classifier = None
            self._matcher_tup = None

    def _update_array(self, new_embs_arr, new_labels_arr):
        '''
        add new embs and labels to current matcher
        :param embs_arr: list of embs
        :param labels_arr: list of label (face id) for each emb
        '''
        if self._matcher_tup is None:
            # no matcher yet, call fit instead
            self._fit_array(new_embs_arr, new_labels_arr)
        else:
            self._rwlock.writer_acquire()
            # fit classifier to new embs
            self._classifier.add(new_embs_arr.astype('float32'))
            # update embs and labels
            old_embs_arr = self._matcher_tup.embs_arr
            old_labels_arr = self._matcher_tup.labels_arr
            embs_arr = np.vstack((old_embs_arr, new_embs_arr))
            labels_arr = np.concatenate((old_labels_arr, new_labels_arr))
            self._matcher_tup = FaissTuple(embs_arr, labels_arr,
                                           embs_arr.shape[0])
            self._rwlock.writer_release()


class SVMMatcher(AbstractMatcher):
    '''
    Classify face id using SVM
    '''

    def match(self,
              emb,
              top_matches=Config.Matcher.MAX_TOP_MATCHES,
              return_min_dist=False):
        '''
        See superclass doc
        '''
        emb = emb.reshape((-1, Config.Matcher.EMB_LENGTH))
        match = self.match_arr(emb, top_matches, return_min_dist)
        if return_min_dist:
            predict_id, top_match_ids, min_dist = match
            return predict_id[0], top_match_ids, min_dist[0]
        else:
            predict_id, top_match_ids = match
            return predict_id[0], top_match_ids

    def match_arr(self,
                  embs_arr,
                  top_matches=Config.Matcher.MAX_TOP_MATCHES,
                  return_min_dist=False):
        if self._matcher_tup is not None:
            self._rwlock.reader_acquire()
            top_matches = min(top_matches, len(self._classifier.classes_) - 1)
            predict_ids = self._classifier.predict(embs_arr)
            probs = self._classifier.predict_proba(embs_arr)
            predict_ids = self._classifier.classes_[np.argmax(probs, axis=1)]
            predict_ids[np.max(probs, axis=1) < Config.Matcher.SVM_PROBS] = 'N'
            print(predict_ids)
            inds = np.argpartition(probs, top_matches)[:top_matches]
            top_track_ids = self._classifier.classes_[inds]
            self._rwlock.reader_release()
        else:
            predict_ids = np.full(embs_arr.shape[0], Config.Matcher.NEW_FACE)
            top_track_ids = np.array([])
            dist = np.full(embs_arr.shape[0], -1)

        if return_min_dist:
            return predict_ids, top_track_ids, dist
        return predict_ids, top_track_ids

    def _fit_array(self, embs_arr, labels_arr):
        length = embs_arr.shape[0]
        if length > 0:
            # only fit if we have data
            self._classifier = SVC(kernel='linear', probability=True)
            self._classifier.fit(embs_arr, labels_arr)
            self._matcher_tup = FaissTuple(embs_arr, labels_arr, length)
        else:
            self._classifier = None
            self._matcher_tup = None

    def _update_array(self, embs_arr, labels_arr):
        if self._matcher_tup is None:
            self._fit_array(embs_arr, labels_arr)
        else:
            old_embs_arr = self._matcher_tup.embs_arr
            old_labels_arr = self._matcher_tup.labels_arr
            embs_arr = np.vstack((old_embs_arr, embs_arr))
            labels_arr = np.concatenate((old_labels_arr, labels_arr))
            self._fit_array(embs_arr, labels_arr)


class HNSWMatcher(AbstractMatcher):
    '''
    Classify face id using HNSWM
    '''

    def match(self, emb, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold=None, return_dists=False):
        emb = emb.reshape((-1, Config.Matcher.EMB_LENGTH))
        match = self.match_arr(emb, top_matches, threshold, True)
        top_match_ids, dists = match
        if return_dists:
            return top_match_ids[0], dists[0]
        else:
            return top_match_ids[0]

    def match_arr(self, embs, top_matches=Config.Matcher.MAX_TOP_MATCHES, \
            threshold=None, return_dists=False):
        # embs (n, Config.Matcher.EMB_LENGTH)
        if threshold is None:
            threshold = self._threshold
        if self._classifier is not None:
            top_matches = min(top_matches, self._matcher_tup.length - 1)
            self._rwlock.reader_acquire()
            inds, dists = self._classifier.knn_query(
                embs.astype('float32'), k=top_matches)
            predict_ids = self._matcher_tup.labels_arr[inds[:, 0]]
            min_dist = dists[:, 0]
            predict_ids[min_dist > self._threshold] = Config.Matcher.NEW_FACE
            top_match_ids = self._matcher_tup.labels_arr[inds]
            self._rwlock.reader_release()
        else:
            top_match_ids = np.full((embs.shape[0], 1), Config.Matcher.NEW_FACE)
            # predict_ids = np.full(embs.shape[0], Config.Matcher.NEW_FACE)
            dists = np.full((embs.shape[0], 1), -1)

        if return_dists:
            return top_match_ids.tolist(), dists.tolist()
        return top_match_ids.tolist()
        # TODO: Merge david logic
        '''
            if min_dist <= threshold:
                top_match_ids = [self._matcher_tup.labels[idx] for (i, idx) in enumerate(inds) if dists[i] <= threshold]
            else:
                if always_return_closest:
                    top_match_ids = [predict_id]
                else:
                    top_match_ids = []
            dists = dists[0:len(top_match_ids)]
        '''

    def _fit_array(self, embs_arr, labels_arr):
        '''
        Fit current matcher to new embs and labels
        :param embs: list of embs
        :param labels: list of label (face id) for each emb
        '''
        length = embs_arr.shape[0]
        if length > 0:
            # only fit if we have data
            self._rwlock.writer_acquire()
            self._classifier = hnswlib.Index(space='l2', dim=Config.Matcher.EMB_LENGTH)
            self._classifier.init_index(max_elements=Config.Matcher.HNSW_MAX_ELEMENTS, ef_construction=200, M=16)
            self._classifier.set_num_threads(Config.Matcher.HNSW_NROF_THREADS)
            self._classifier.add_items(embs_arr.astype('float32'), np.arange(labels_arr.shape[0]))
            self._classifier.set_ef(labels_arr.shape[0]+10)
            self._matcher_tup = FaissTuple(embs_arr, labels_arr, length)
            self._rwlock.writer_release()
        else:
            self._classifier = None
            self._matcher_tup = None

    def _update_array(self, new_embs_arr, new_labels_arr):
        '''
        add new embs and labels to current matcher
        :param embs_arr: list of embs
        :param labels_arr: list of label (face id) for each emb
        '''
        if self._matcher_tup is None:
            # no matcher yet, call fit instead
            self._fit_array(new_embs_arr, new_labels_arr)
        else:
            self._rwlock.writer_acquire()
            old_embs_arr = self._matcher_tup.embs_arr
            old_labels_arr = self._matcher_tup.labels_arr
            # fit classifier to new embs
            self._classifier.add_items(new_embs_arr.astype('float32'), np.arange(old_labels_arr.shape[0], \
                                            old_labels_arr.shape[0] + new_labels_arr.shape[0]))
            # update embs and labels
            embs_arr = np.vstack((old_embs_arr, new_embs_arr))
            labels_arr = np.concatenate((old_labels_arr, new_labels_arr))
            self._matcher_tup = FaissTuple(embs_arr, labels_arr,
                                           embs_arr.shape[0])
            self._rwlock.writer_release()
