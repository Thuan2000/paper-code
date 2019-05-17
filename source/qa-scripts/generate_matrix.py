from matcher import FaissMatcher
import json
import os
import argparse
from collections import Counter
from config import Config
from scipy import misc
from random import shuffle
from tf_graph import FaceGraph
from face_extractor import FacenetExtractor
from preprocess import Preprocessor, align_and_crop
from cv_utils import create_if_not_exist, PickleUtils, CropperUtils
import numpy as np
import csv
from pymongo import MongoClient

extractor_graph = FaceGraph()
face_extractor = FacenetExtractor(extractor_graph)
preprocessor = Preprocessor()

mongodb_client = MongoClient(
    Config.MongoDB.IP_ADDRESS,
    Config.MongoDB.PORT,
    username=Config.MongoDB.USERNAME,
    password=Config.MongoDB.PASSWORD)

mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
mongodb_dashinfo = mongodb_db[Config.MongoDB.DASHINFO_COLS_NAME]
mongodb_faceinfo = mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]

mrate_learning_rate = (0.1, 0.1, 0.9)  #major rate
vrate_learning_rate = (0.1, 0.1, 0.9)  #valid rate
dist_learning_rate = (0.2, 0.01, 0.8)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def merge_reg_dict(regdict1, regdict2):
    regdict_ = regdict1.copy()
    for item in regdict2:
        if item not in regdict_:
            regdict_[item] = regdict2[item].copy()
    return regdict_


def get_top_ids_list(predicted_ids, predicted_dists):
    id_list = Counter(predicted_ids).most_common()
    l_top_ids = []
    l_dists = []
    for (i, dist) in enumerate(predicted_dists):
        if predicted_ids[i] not in l_top_ids:
            l_top_ids.append(predicted_ids[i])
            l_dists.append(dist)
    return l_top_ids, l_dists


def merge_folder_struct(folder_struct1, folder_struct2):
    merged_struct = folder_struct1.copy()
    for label in folder_struct2:
        if label not in merged_struct:
            merged_struct[label] = {}
        for tracker in folder_struct2[label]:
            if tracker not in merged_struct[label]:
                merged_struct[label][tracker] = []
            merged_struct[label][tracker] += folder_struct2[label][tracker]
    return merged_struct


def get_score(ids_f1):
    p_count = 0
    r_count = 0
    precision = 0
    recall = 0
    for record in ids_f1.values():
        if record[1] + record[0] > 0:  #TP + FN
            recall += record[0] / (record[1] + record[0])
            r_count += 1
        if record[0] + record[3] > 0:
            precision += record[0] / (record[0] + record[3])
            p_count += 1
    precision = precision / p_count
    recall = recall / r_count
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def predict_tracker_id(major_rate, n_valid_element_rate, distance_threshold,
                       predicted_ids, predicted_dists):
    n_valid_elements = 0
    nof_predicted = 0
    if (len(predicted_ids) > 0):
        final_predict_id, nof_predicted = Counter(predicted_ids).most_common(
            1)[0]
        for i, id in enumerate(predicted_ids):
            if id == final_predict_id and predicted_dists[
                    i] <= distance_threshold:
                n_valid_elements += 1

        #verify by major_rate and valid element rate
        if nof_predicted / len(predicted_ids) < major_rate \
          or n_valid_elements / nof_predicted < n_valid_element_rate:
            final_predict_id = Config.Matcher.NEW_FACE
    else:
        final_predict_id = Config.Matcher.NEW_FACE

    return final_predict_id


def build_matcher(data_struct):
    matcher = FaissMatcher()
    embs = []
    labels = []
    for label in data_struct["folder_structure"]:
        for tracker in data_struct["folder_structure"][label]:
            for image_id in data_struct["folder_structure"][label][tracker]:
                labels.append(image_id)
                embs.append(data_struct["regdict"][image_id]["embedding"])
    matcher.fit(embs, labels)  #build matcher
    return matcher


def load_embeddings_from_path(data_path, json_save_path, from_mongo=True):
    try:
        with open(json_save_path, "r") as f:
            data_structure = json.loads(f.read())
    except:
        data_structure = {"folder_structure": {}, "regdict": {}}
        folder_structure = data_structure["folder_structure"]
        reg_dict = data_structure["regdict"]
        '''
		print folder_structure -->
		{
			label: {
					trackid1 : [imageid1, imageid2,..]
					...
			}
		}
		print regdict ---->
		{
			imageid: {
					"face_id" : label
					"embedding" : [emb_array]
			}
		}
		'''
        labels = [
            folder for folder in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, folder))
        ]
        for label in labels:
            trackers = [os.path.join(data_path,label,tracker) for tracker in os.listdir(os.path.join(data_path,label)) \
                           if os.path.isdir(os.path.join(data_path,label,tracker))]
            folder_structure[label] = {}
            for tracker in trackers:
                folder_structure[label][tracker] = []
                image_paths = [
                    os.path.join(tracker, img_path)
                    for img_path in os.listdir(tracker)
                    if "jpg" in img_path
                ]
                for img_path in image_paths:
                    img = misc.imread(os.path.join(img_path))
                    image_id = img_path.split('/')[-1].replace(
                        ".jpg", "")  #get image id
                    data_split = image_id.split('_')
                    padded_bbox = data_split[-4:len(data_split)]
                    padded_bbox = '_'.join(padded_bbox)
                    # time_stamp = float(data_split[5])
                    cropped_face = CropperUtils.reverse_display_face(
                        img, padded_bbox)
                    preprocessed_image = preprocessor.process(cropped_face)

                    if from_mongo:
                        try:
                            print(mongodb_faceinfo.find_one({
                                "image_id": image_id
                            }))
                            emb_array = np.asarray(
                                mongodb_faceinfo.find_one({
                                    "image_id": image_id
                                })["embedding"])
                            print("Load from mongodb")
                        except:
                            emb_array, _ = face_extractor.extract_features(
                                preprocessed_image)
                    else:
                        print("extract feature normally")
                        emb_array, _ = face_extractor.extract_features(
                            preprocessed_image)
                    print(image_id)
                    folder_structure[label][tracker].append(image_id)
                    reg_dict[image_id] = {
                        "face_id": label,
                        "embedding": emb_array
                    }
        data_file = open(json_save_path, "w")
        data_file.write(json.dumps(data_structure, cls=NumpyEncoder))
        data_file.close()
    return data_structure


def load_matching_list(train_struct,
                       evaluate_struct,
                       output_file,
                       top_matches=10,
                       distance_threshold=1.0):
    try:
        with open(output_file, "r") as f:
            eval_match_list = json.loads(f.read())
            return eval_match_list
    except:
        #build faiss matcher from our datastructure
        matcher = build_matcher(train_struct)

        eval_set = evaluate_struct["folder_structure"]
        eval_dict = evaluate_struct["regdict"]
        eval_match_list = {}

        for groundtruth in eval_set:
            label = groundtruth
            if groundtruth not in train_struct["folder_structure"]:
                label = Config.Matcher.NEW_FACE
            if label not in eval_match_list:
                eval_match_list[label] = {}
            for tracker in eval_set[groundtruth]:
                eval_match_list[label][tracker] = {}
                for image_id in eval_set[groundtruth][tracker]:
                    top_ids, dists = matcher.match(eval_dict[image_id]["embedding"], threshold = distance_threshold,top_matches = top_matches,\
                             return_dists=True, always_return_closest = True)
                    eval_match_list[label][tracker][image_id] = [[top_ids[i], dists[i]] for i \
                                   in range(len(top_ids))]
                    print("Generated match list for: " + image_id)
        data_file = open(output_file, "w")
        data_file.write(json.dumps(eval_match_list, cls=NumpyEncoder))
        data_file.close()
        return eval_match_list


def evaluate(match_dict, regdict, train_struct, params):
    n_wrong_pairs = 0
    l_wrong_pairs = []
    ids_f1 = {}
    for id in train_struct["folder_structure"].keys():
        ids_f1[id] = [0, 0, 0, 0]  #TP, FN, #TN, FP
    ids_f1["NEW_FACE"] = [0, 0, 0, 0]
    major_rate, n_valid_element_rate, distance_threshold = params
    for groundtruth in match_dict:
        for tracker in match_dict[groundtruth]:
            predicted_ids = []
            predicted_dists = []
            for image_id in match_dict[groundtruth][tracker]:
                shorten_dists = [item for item in match_dict[groundtruth][tracker][image_id] if \
                     item[1] <= distance_threshold]
                if len(shorten_dists) > 0:
                    top_ids, dists = [i[0] for i in shorten_dists], [
                        i[1] for i in shorten_dists
                    ]
                else:
                    top_ids, dists = [match_dict[groundtruth][tracker][image_id][0][0]], \
                        [match_dict[groundtruth][tracker][image_id][0][1]]
                predicted_ids += top_ids
                predicted_dists += dists

            #convert to faceid
            predicted_ids = [regdict[id]["face_id"] for id in predicted_ids]
            #predicted_ids, predicted_dists = get_top_ids_list(predicted_ids, predicted_dists)

            final_predict_id = predict_tracker_id(major_rate, n_valid_element_rate, distance_threshold,\
                         predicted_ids, predicted_dists)

            if final_predict_id != groundtruth:
                ids_f1[groundtruth][1] += 1  #FN
                ids_f1[final_predict_id][3] += 1  #FP
                n_wrong_pairs += 1
                l_wrong_pairs.append((groundtruth, final_predict_id,
                                      predicted_ids, predicted_dists))
            else:
                ids_f1[groundtruth][0] += 1  #TP

    return get_score(ids_f1), ids_f1


def tune_threshold(match_dict, regdict, train_struct):
    l_hyper_params = []
    m_rate = mrate_learning_rate[0]
    while m_rate <= mrate_learning_rate[2]:
        v_rate = vrate_learning_rate[0]
        while v_rate <= vrate_learning_rate[2]:
            dist_rate = dist_learning_rate[0]
            while dist_rate <= dist_learning_rate[2]:
                l_hyper_params.append((m_rate, v_rate, dist_rate))
                dist_rate += dist_learning_rate[1]
            v_rate += vrate_learning_rate[1]
        m_rate += mrate_learning_rate[1]

    results = []
    results_ = []
    for i, params in enumerate(l_hyper_params):
        score, ids_f1 = evaluate(match_dict, regdict, train_struct, params)
        results.append(score)
        #results_.append(l_wrong_pairs)

    best_result = np.max([score[2] for score in results])
    best_params = [
        i for i in range(len(l_hyper_params)) if results[i][2] == best_result
    ]

    return l_hyper_params, results, best_params


def evaluate_on_test(train_struct, test_truct, params, top_matches=10):
    major_rate, n_valid_element_rate, distance_threshold = params

    matcher = build_matcher(train_struct)
    folder_structure = test_truct["folder_structure"]
    reg_dict = merge_reg_dict(train_struct["regdict"], test_truct["regdict"])
    acceptable_new_face = []
    wrong_pairs = 0

    #rule out non-union labels
    for label in folder_structure:
        if label not in train_struct["folder_structure"]:
            acceptable_new_face.append(label)
    ids_f1 = {}
    for id in train_struct["folder_structure"].keys():
        ids_f1[id] = [0, 0, 0, 0]  #TP, FN, #TN, FP
    ids_f1["NEW_FACE"] = [0, 0, 0, 0]
    for groundtruth in folder_structure:

        for tracker in folder_structure[groundtruth]:
            images_list = folder_structure[groundtruth][tracker]
            elements = [
                reg_dict[image_id]["embedding"] for image_id in images_list
            ]
            predicted_ids = []
            predicted_dists = []
            for element in elements:
                matched_ids, dists = matcher.match(element, threshold = distance_threshold, top_matches = top_matches,\
                          return_dists=True, always_return_closest = True)
                predicted_ids += matched_ids
                predicted_dists += dists

            predicted_ids = [reg_dict[id]["face_id"] for id in predicted_ids]

            #rule out repeated ids, only return list of top predicted ids
            #predicted_ids, predicted_dists = get_top_ids_list(predicted_ids, predicted_dists)

            final_predict_id = predict_tracker_id(major_rate, n_valid_element_rate, distance_threshold,\
                          predicted_ids, predicted_dists)

            if final_predict_id != groundtruth:
                if groundtruth not in ids_f1:
                    ids_f1[groundtruth] = [0, 0, 0, 0]
                if final_predict_id not in ids_f1:
                    ids_f1[final_predict_id] = [0, 0, 0, 0]

                if final_predict_id != Config.Matcher.NEW_FACE:
                    if groundtruth not in acceptable_new_face:
                        ids_f1[groundtruth][1] += 1
                    else:
                        ids_f1[Config.Matcher.NEW_FACE][1] += 1

                    ids_f1[final_predict_id][3] += 1
                    wrong_pairs += 1
                else:
                    if groundtruth not in acceptable_new_face:
                        ids_f1[groundtruth][1] += 1
                        ids_f1[final_predict_id][3] += 1
                        wrong_pairs += 1
                    else:
                        #acceptable_new_face.remove(groundtruth)
                        new_embs = elements
                        new_labels = images_list
                        matcher.update(new_embs, new_labels)
                        ids_f1[groundtruth] = [0, 0, 0, 0]
                        ids_f1[final_predict_id][0] += 1
                # if groundtruth not in acceptable_new_face:
                # 	ids_f1[groundtruth][1] += 1
                # 	ids_f1[final_predict_id][3] += 1
                # 	wrong_pairs += 1
                # 	#print((groundtruth, final_predict_id, predicted_ids,predicted_dists))
                # else:
                # 	if(final_predict_id != Config.Matcher.NEW_FACE):
                # 		acceptable_new_face.remove(groundtruth)
                # 		new_embs = elements
                # 		new_labels = images_list
                # 		matcher.update(new_embs, new_labels)
                # 		ids_f1[groundtruth] = [0,0,0,0]
                # 		ids_f1[final_predict_id][0] += 1
                # 	else:
                # 		ids_f1[final_predict_id][3] += 1
            else:
                new_embs = elements
                new_labels = images_list
                matcher.update(new_embs, new_labels)
                ids_f1[groundtruth][0] += 1

    return get_score(ids_f1), ids_f1


def f1_matrix_to_csv(hyper_params, ids_f1, output):
    with open(output, 'w') as csvfile:
        spamwriter = csv.writer(
            csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['hyper_params', 'ID', 'TP', 'FN', 'TN', 'FP'])
        for i, params in enumerate(hyper_params):
            params = ' '.join([str(i) for i in params])
            spamwriter.writerow([params])
            for label in ids_f1[i]:
                if (sum(ids_f1[i][label]) > 0):
                    spamwriter.writerow([
                        '', label, ids_f1[i][label][0], ids_f1[i][label][1],
                        ids_f1[i][label][2], ids_f1[i][label][3]
                    ])


def match_list_to_csv(match_list):
    with open('match_list.csv', 'w') as csvfile:
        spamwriter = csv.writer(
            csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(
            ['ground_truth', 'Tracker', 'imageid', 'top_matches -->'])
        for label in match_list:
            spamwriter.writerow(label)
            for tracker in match_list[label]:
                spamwriter.writerow(['', tracker.split("/")[-1]])
                for image_id in match_list[label][tracker]:
                    this_row = ['', '', image_id]
                    for match in match_list[label][tracker][image_id]:
                        this_row.append(match[0])
                        this_row.append(match[1])
                    spamwriter.writerow(this_row)
    print("write match_list to csv")


def hyper_param_to_csv(l_hyper_params, results, output_file):
    with open(output_file, 'w') as csvfile:
        spamwriter = csv.writer(
            csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Hyper params', 'Precision', 'Recall', 'F1'])
        for i, params in enumerate(l_hyper_params):
            params = [str(i) for i in params]
            spamwriter.writerow([
                ' '.join(params),
                str(results[i][0]),
                str(results[i][1]),
                str(results[i][2])
            ])
    print("write hyper_param to csv")


def create_validate_from_train(train_struct, train_ratio):
    validate_struct = {"folder_structure": {}, "reg_dict": {}}

    n_train_struct = {"folder_structure": {}, "reg_dict": {}}

    for label in train_struct["folder_structure"]:
        validate_struct["folder_structure"][label] = {}
        n_train_struct["folder_structure"][label] = {}
        l_trackers = list(train_struct["folder_structure"].keys())
        n_train = len(l_trackers) * train_ratio
        n_validate = len(l_trackers) - n_train
        for trackid in l_trackers[0:n_train]:
            n_train_struct["folder_structure"][label][trackid] = train_struct[
                "folder_structure"][trackid]
            for image_id in train_struct["folder_structure"][trackid]:
                n_train_struct["regdict"][imageid] = train_struct["regdict"][
                    imageid]

        for trackid in l_trackers[-n_validate:]:
            n_train_struct["folder_structure"][label][trackid] = train_struct[
                "folder_structure"][trackid]
            for image_id in train_struct["folder_structure"][trackid]:
                n_train_struct["regdict"][imageid] = train_struct["regdict"][
                    imageid]
    return n_train_struct, validate_struct


def main(train_path, validate_path, test_path, db, prefix):
    train_data_json = train_path.replace("/", "") + "_" + prefix + ".json"
    validate_data_json = validate_path.replace("/", "") + "_" + prefix + ".json"
    test_data_json = test_path.replace("/", "") + "_" + prefix + ".json"
    train_struct = load_embeddings_from_path(
        train_path, train_data_json, from_mongo=db)
    if validate_path is not None:
        validate_struct = load_embeddings_from_path(
            validate_path, validate_data_json, from_mongo=db)
    else:
        train_struct, validate_struct = create_validate_from_train(
            train_struct, 0.4)

    match_dict = load_matching_list(train_struct, validate_struct,
                                    "train_validate_match_" + prefix + ".json")
    match_list_to_csv(match_dict)
    l_hyper_params, results, best_params_indices = tune_threshold(
        match_dict, train_struct["regdict"], train_struct)
    hyper_param_to_csv(l_hyper_params, results, "train_validate_results.csv")
    test_struct = load_embeddings_from_path(
        test_path, test_data_json, from_mongo=db)

    train_validate_struct = {
        "folder_structure":
        merge_folder_struct(train_struct["folder_structure"],
                            validate_struct["folder_structure"]),
        "regdict":
        merge_reg_dict(train_struct["regdict"], validate_struct["regdict"])
    }
    test_match_dict = load_matching_list(train_validate_struct, test_struct,
                                         "test_match_dict_" + prefix + ".json")
    train_test_regdict = merge_reg_dict(train_validate_struct["regdict"],
                                        test_struct["regdict"])
    l_hyper_params, results, best_params_indices = tune_threshold(
        test_match_dict, train_test_regdict, train_validate_struct)
    hyper_param_to_csv(l_hyper_params, results, "train_test_results.csv")

    # print("Test one test set")
    results = []
    f1_matrices = []
    best_params = [l_hyper_params[index] for index in best_params_indices]
    for params in best_params:
        score, f1_matrix = evaluate(test_match_dict, train_test_regdict,
                                    train_validate_struct, params)
        f1_matrices.append(f1_matrix)
        results.append(score)
    f1_matrix_to_csv(best_params, f1_matrices, "f1_matrices.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mrate', '--major_rate', help='', default=0.3)
    parser.add_argument('-vrate', '--valid_rate', help='', default=0.5)

    parser.add_argument(
        '-dist', '--dist', help='', default=Config.Matcher.MIN_ASSIGN_THRESHOLD)

    parser.add_argument('-train', '--train', help='', action='store_true')

    parser.add_argument('-train_path', '--train_path', help='', default=None)

    parser.add_argument(
        '-db', '--db', help='build dict from mongodb', action='store_true')

    parser.add_argument(
        '-validate_path', '--validate_path', help='', default=None)

    parser.add_argument('-test_path', '--test_path', help='', default=None)

    parser.add_argument('-shuffle', '--shuffle', help='', action='store_true')

    parser.add_argument('-prefix', '--prefix', help='', default="1")

    args = parser.parse_args()

    main(args.train_path, args.validate_path, args.test_path, args.db,
         args.prefix)
