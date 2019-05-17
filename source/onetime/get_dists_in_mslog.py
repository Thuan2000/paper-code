from config import Config
from pymongo import MongoClient
import json


def main():
    mongodb_client = MongoClient(
        Config.MongoDB.IP_ADDRESS,
        Config.MongoDB.PORT,
        username=Config.MongoDB.USERNAME,
        password=Config.MongoDB.PASSWORD)

    mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
    mongodb_dashinfo = mongodb_db[Config.MongoDB.DASHINFO_COLS_NAME]
    mongodb_faceinfo = mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]
    mongodb_mslog = mongodb_db[Config.MongoDB.MSLOG_COLS_NAME]

    cursors = mongodb_mslog.find({})
    merge_dists = []
    split_dists = []
    for cursor in cursors:
        tmp_dict = {}
        image_id = cursor['image_id']
        tmp_dict['predicted_id'] = cursor['old_face_id']
        dash_cursors = mongodb_dashinfo.find({'represent_image_id': image_id})
        dash_cursor = dash_cursors[0]
        tmp_dict['track_id'] = dash_cursor['track_id']
        tmp_dict['recognized_type'] = dash_cursor['recognized_type']
        tmp_dict['matching_result_dict'] = json.loads(
            dash_cursor['matching_result'])
        if not tmp_dict['predicted_id'] in tmp_dict['matching_result_dict']:
            tmp_dict['predicted_dist'] = tmp_dict['matching_result_dict'][
                Config.Matcher.NEW_FACE]['dist']
            tmp_dict['tag_face_type'] = Config.Matcher.NEW_FACE
            tmp_dict['rate'] = tmp_dict['matching_result_dict'][
                Config.Matcher.NEW_FACE]['rate']
        else:
            tmp_dict['predicted_dist'] = tmp_dict['matching_result_dict'][
                tmp_dict['predicted_id']]['dist']
            tmp_dict['tag_face_type'] = 'OLD_FACE'
            tmp_dict['rate'] = tmp_dict['matching_result_dict'][
                tmp_dict['predicted_id']]['rate']
        if cursor['action_type'] == 'merge':
            merge_dists.append(tmp_dict)
        else:
            split_dists.append(tmp_dict)

    print('Merge list:')
    for element in merge_dists:
        print('Track ID: {}\n'
              'Matching Results: {}\n'
              'Predict ID: {} - {}\n'
              'Dist - Rate: {} - {}\n\n'.format(
                  element['track_id'], element['matching_result_dict'],
                  element['predicted_id'], element['tag_face_type'],
                  element['predicted_dist'], element['rate']))
    print('Split list:')
    for element in split_dists:
        print('Track ID: {}\n'
              'Matching Results: {}\n'
              'Predict ID: {} - {}\n'
              'Dist - Rate: {} - {}\n\n'.format(
                  element['track_id'], element['matching_result_dict'],
                  element['predicted_id'], element['tag_face_type'],
                  element['predicted_dist'], element['rate']))


if __name__ == '__main__':
    main()
