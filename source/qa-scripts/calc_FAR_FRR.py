import os
import sys
import time
import pickle
import random
import argparse
import itertools
import numpy as np
import pylab as pl
import pandas as pd
from cv_utils import l2_distance, sum_square_distance, Features


class Result:

    def __init__(self, threshold, tp, fp, tn, fn):
        self.threshold = threshold
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.FAR = fp / (fp + tn + 1e-6)
        self.FRR = fn / (fn + tp + 1e-6)
        self.precision = tp / (tp + fp + 1e-6)
        self.recall = tp / (tp + fn + 1e-6)


class Evaluation:
    # def __init__(self, embs, ids):
    #     self.embs = embs
    #     self.ids = ids

    def load_data(self, embs_ids_file, new_data=None):
        self.features = Features(embs_ids_file)
        if new_data:
            self.new_model_features = Features(new_data)

    def generate_id_emb_index_dict(self, limit_nrof_faces):
        id_face_dict = {}
        for i in range(self.features.get_nrof_embs()):
            _id = self.features.get_id(i)
            if _id in id_face_dict:
                id_face_dict[_id].append(i)
            else:
                id_face_dict[_id] = [i]

        for key, value in id_face_dict.items():
            current_len = len(value)
            nrof_faces_to_compare = min(current_len, limit_nrof_faces)
            id_face_dict[key] = random.sample(value, nrof_faces_to_compare)
        return id_face_dict

    def generate_compare_pairs(self, nrof_pairs, limit_nrof_faces, random_seed):
        pos_pairs = []
        neg_pairs = []
        id_face_dict = self.generate_id_emb_index_dict(limit_nrof_faces)
        ids = list(id_face_dict.keys())
        nrof_id = len(ids)
        for idx in range(nrof_id):
            this_id = ids[idx]
            this_id_faces = id_face_dict[this_id]
            # get positive pairs from faces of this id
            this_pos_pairs = list(itertools.combinations(this_id_faces, r=2))
            pos_pairs += this_pos_pairs
            # get negative pairs by product all this id faces to other ids faces
            other_ids_faces = []
            for k in ids[idx + 1:]:
                other_ids_faces += id_face_dict[k]
            # only generate nrof neg pairs equal to nrof pos pairs
            if other_ids_faces:
                this_neg_pairs = []
                for i in range(len(this_pos_pairs)):
                    this_neg_pairs.append((random.choice(other_ids_faces),
                                           random.choice(this_id_faces)))
                neg_pairs += this_neg_pairs

        min_nrof_pairs = min(len(pos_pairs), len(neg_pairs))
        nrof_pairs = min(nrof_pairs, min_nrof_pairs)
        return_pos_pairs = random.sample(pos_pairs, nrof_pairs)
        return_neg_pairs = random.sample(neg_pairs, nrof_pairs)

        print('number of pair: ', nrof_pairs)
        return return_pos_pairs, return_neg_pairs

    def get_dist_actual_issame_array(self, pos_pairs, neg_pairs, distance='l2'):
        dists = []
        actual_issame = []
        # if input two embs file from two folders
        if hasattr(self, 'new_model_features'):
            for (emb_idx_1, emb_idx_2) in pos_pairs:
                emb_1a = self.features.get_emb(emb_idx_1)
                emb_2a = self.features.get_emb(emb_idx_2)
                emb_1b = self.new_model_features.get_emb(emb_idx_1)
                emb_2b = self.new_model_features.get_emb(emb_idx_2)
                emb_1 = np.vstack([emb_1a, emb_1b])
                emb_2 = np.vstack([emb_2a, emb_2b])
                dist = self.__get_distance(emb_1, emb_2, distance)
                dists.append(dist)
                actual_issame.append(True)

            for (emb_idx_1, emb_idx_2) in neg_pairs:
                emb_1a = self.features.get_emb(emb_idx_1)
                emb_2a = self.features.get_emb(emb_idx_2)
                emb_1b = self.new_model_features.get_emb(emb_idx_1)
                emb_2b = self.new_model_features.get_emb(emb_idx_2)
                emb_1 = np.vstack([emb_1a, emb_1b])
                emb_2 = np.vstack([emb_2a, emb_2b])
                dist = self.__get_distance(emb_1, emb_2, distance)
                dists.append(dist)
                actual_issame.append(False)

            return dists, actual_issame

        # else do normal, only take 1 embs data
        for (emb_idx_1, emb_idx_2) in pos_pairs:
            emb_1 = self.features.get_emb(emb_idx_1)
            emb_2 = self.features.get_emb(emb_idx_2)
            dist = self.__get_distance(emb_1, emb_2, distance)
            dists.append(dist)
            actual_issame.append(True)

        for (emb_idx_1, emb_idx_2) in neg_pairs:
            emb_1 = self.features.get_emb(emb_idx_1)
            emb_2 = self.features.get_emb(emb_idx_2)
            dist = self.__get_distance(emb_1, emb_2, distance)
            dists.append(dist)
            actual_issame.append(False)

        return dists, actual_issame

    def find_all_tp_fp(self, dists, actual_issame):
        np_dists = np.array((dists))
        sort_index = np_dists.argsort()
        sorted_dists = [dists[i] for i in sort_index]
        sorted_actual_issame = np.array(
            ([actual_issame[i] for i in sort_index]))
        nrof_pos_pairs = sorted_actual_issame.sum()
        nrof_neg_pairs = sorted_actual_issame.size - nrof_pos_pairs

        result_array = []
        for idx, threshold in enumerate(sorted_dists):
            tp, fp = self.__count_tp_fp(sorted_actual_issame[:idx])
            tn = nrof_neg_pairs - fp
            fn = nrof_pos_pairs - tp
            result_array.append(Result(threshold, tp, fp, tn, fn))
        return result_array

    def result_with_input_threshold(self, dists, actual_issame, threshold):
        np_dists = np.array((dists))
        sort_index = np_dists.argsort()
        sorted_dists = np.array([dists[i] for i in sort_index])
        sorted_actual_issame = np.array(
            ([actual_issame[i] for i in sort_index]))
        nrof_pos_pairs = sorted_actual_issame.sum()
        nrof_neg_pairs = sorted_actual_issame.size - nrof_pos_pairs
        idx = np.where(sorted_dists < threshold)[0][-1]
        tp, fp = self.__count_tp_fp(sorted_actual_issame[:idx])
        tn = nrof_neg_pairs - fp
        fn = nrof_pos_pairs - tp
        return Result(threshold, tp, fp, tn, fn)

    def plot(self, results):
        thresholds = []
        far = []
        frr = []

        for result in results:
            thresholds.append(result.threshold)
            far.append(result.FAR)
            frr.append(result.FRR)

        far = np.array((far))
        frr = np.array((frr))
        thresholds = np.array((thresholds))

        i = np.arange(len(far))  # index for df
        roc = pd.DataFrame({
            'thresholds': pd.Series(thresholds, index=i),
            'far': pd.Series(far, index=i),
            'frr': pd.Series(frr, index=i)
        })
        roc.ix[(roc.frr - 0).abs().argsort()[:1]]

        # Plot far,frr
        fig, ax = pl.subplots()
        pl.plot(roc['far'])
        pl.plot(roc['frr'], color='red')
        pl.xlabel('Thresholds')
        pl.ylabel('FAR(blue), FRR(red)')
        pl.title('Receiver operating characteristic')
        ax.set_xticklabels([])
        pl.show()

    def __count_tp_fp(self, actual_issame):
        tp = actual_issame.sum()
        fp = actual_issame.size - tp
        return tp, fp

    def __get_distance(self, emb_1, emb_2, distance):
        if 'l2' in distance:
            return l2_distance(emb_1, emb_2)
        else:
            return sum_square_distance(emb_1, emb_2)


def main(args):
    # test_dict = {'a':[1,2,3], 'b':[4,5]}
    evaluator = Evaluation()
    print('loading data')
    evaluator.load_data(args.data, args.new_data)
    print('generating compare pairs')
    pos_pairs, neg_pairs = evaluator.generate_compare_pairs(
        args.nrof_pairs, args.limit_nrof_faces, args.random_seed)
    print('computing distances')
    dists, actual_issame = evaluator.get_dist_actual_issame_array(
        pos_pairs, neg_pairs)
    if args.threshold:
        # run in test with fixed threshold mode
        result = evaluator.result_with_input_threshold(dists, actual_issame,
                                                       args.threshold)
        print('precision = {} at recall = {}'.format(result.precision,
                                                     result.recall))
        print('FAR = {} at FRR = {}'.format(result.FAR, result.FRR))

    else:
        # vary threshold to find interesting points
        print('varying thresholds')
        results = evaluator.find_all_tp_fp(dists, actual_issame)
        if args.save_results:
            save_name = os.path.splitext(os.path.basename(
                args.data))[0] + '_' + str(time.time()) + '.pkl'
            save_path = os.path.join(os.getcwd(), save_name)
            pickle.dump(results, open(save_path, 'wb'))
            print('result saved at', save_path)
        FRRs = np.array([result.FRR for result in results])
        FARs = np.array([result.FAR for result in results])
        precisions = np.array([result.precision for result in results])
        recalls = np.array([result.recall for result in results])
        FRR_01 = np.argmin(abs(FRRs - 0.1))
        FRR_001 = np.argmin(abs(FRRs - 0.01))
        nearest_EER = np.argmin(abs(FARs - FRRs))
        precision_09 = np.argmin(abs(precisions - 0.9))
        recall_07 = np.argmin(abs(recalls - 0.7))

        _far = round(results[FRR_01].FAR, 4)
        _frr = round(results[FRR_01].FRR, 4)
        _threshold = round(results[FRR_01].threshold, 4)
        print('FAR = {} at FRR = {}, threshold = {}'.format(
            _far, _frr, _threshold))
        _far = round(results[FRR_001].FAR, 4)
        _frr = round(results[FRR_001].FRR, 4)
        _threshold = round(results[FRR_001].threshold, 4)
        print('FAR = {} at FRR = {}, threshold = {}'.format(
            _far, _frr, _threshold))
        _eer = round(results[nearest_EER].FAR, 4)
        _threshold = round(results[nearest_EER].threshold, 4)
        print('Nearest EER value = {}, threshold = {}'.format(_eer, _threshold))
        _prec = round(results[precision_09].precision, 4)
        _recall = round(results[precision_09].recall, 4)
        _threshold = round(results[precision_09].threshold, 4)
        print('precision = {} at recall = {}, threshold = {}'.format(
            _prec, _recall, _threshold))
        _prec = round(results[recall_07].precision, 4)
        _recall = round(results[recall_07].recall, 4)
        _threshold = round(results[recall_07].threshold, 4)
        print('precision = {} at recall = {}, threshold = {}'.format(
            _prec, _recall, _threshold))
    if args.plot:
        evaluator.plot(results)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        'parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data', help='path to data file contain embeddings and labels')
    parser.add_argument('--new_data', help='path to data from another model')
    parser.add_argument(
        '--threshold',
        help='threshold for one time run results',
        type=float,
        default=None)
    parser.add_argument(
        '--plot',
        help='decide either or not to plot the far, frr graph',
        action='store_true')
    parser.add_argument(
        '--nrof_pairs',
        help='number of positive, negative pairs',
        type=int,
        default=999999999999)
    parser.add_argument(
        '--distance',
        help='choose the distance to be used: l2/sum-square',
        default='l2')
    parser.add_argument(
        '--limit_nrof_faces',
        help='only take x face for each id to compare',
        default=99999999999,
        type=int)
    parser.add_argument(
        '--random_seed',
        help='set the seed for randoming pairs',
        default=0,
        type=int)
    parser.add_argument(
        '--verbose', '-v', help='print log lines', action='store_true')
    parser.add_argument(
        '--save_results',
        '-sr',
        help='save all tpfp found',
        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
