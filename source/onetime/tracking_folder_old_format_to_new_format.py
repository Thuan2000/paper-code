import glob
import os
import shutil
import subprocess
import argparse
from cv_utils import create_if_not_exist, PickleUtils


def get_history_in_send_folder():
    his_dic = {}
    send_rbmq_dir = '/home/manho/source-code/iq_facial_recognition/data/send_rbmq'
    img_dirs = glob.glob(send_rbmq_dir + '/*.jpg')
    for img_dir in img_dirs:
        print("Processing " + img_dir)
        img_mtime = os.stat(img_dir).st_mtime
        file_name = img_dir.split('/')[-1].split('.')[0]
        splited_file_name = file_name.split('_')
        face_id = splited_file_name[0]
        if not face_id in list(his_dic.keys()):
            his_dic[face_id] = []
        his_dic[face_id].append(img_mtime)
        his_dic[face_id] = sorted(his_dic[face_id])
    print("Saved history dictionary pickle!")
    PickleUtils.save_pickle('/home/manho/data/his_dic.pkl', his_dic)


def modify_image_id(img_path, track_id, time_stamp=None):
    img_dirs = glob.glob(img_path + '/*.jpg')
    print("Modifying name of {} files".format(len(img_dirs)))
    for img_dir in img_dirs:
        splited_img_dir = img_dir.split('/')
        print(splited_img_dir)
        file_name = splited_img_dir[-1].replace('.jpg', '')
        print(file_name)
        splited_file_name = file_name.split('_')
        print(splited_file_name)
        if time_stamp is not None:
            splited_file_name[5] = str(time_stamp)
        splited_file_name.append(str(track_id))
        new_file_name = '_'.join(splited_file_name)
        ext_new_file_name = new_file_name + '.jpg'
        splited_img_dir[-1] = ext_new_file_name
        dst_dir = os.path.join(splited_img_dir)
        os.rename(img_dir, dst_dir)


def main(src, dst):
    # his_dic = get_history_in_send_folder()
    his_dic = PickleUtils.read_pickle('/home/manho/data/his_dic.pkl')
    NEW_TRACKING_DIR = dst
    create_if_not_exist(NEW_TRACKING_DIR)
    track_id_dirs = glob.glob(src + '/*')

    for track_id_dir in track_id_dirs:
        print('Processing ' + track_id_dir)
        splited_file_name = glob.glob(track_id_dir +
                                      '/*')[0].split('/')[-1].replace(
                                          '.jpg', '').split('_')
        face_id = splited_file_name[0]
        track_id = track_id_dir.split('/')[-1]
        print('FACEID: {}, TRACKID: {}'.format(face_id, track_id))

        face_id_dir = os.path.join(NEW_TRACKING_DIR, face_id)
        create_if_not_exist(face_id_dir)

        new_track_id_dir = os.path.join(face_id_dir, track_id)

        subprocess.call(["cp", "-r", track_id_dir, face_id_dir])
        this_mtime = -1
        if face_id in list(his_dic.keys()):
            this_mtime = his_dic[face_id].pop(0)
            if his_dic[face_id] == []:
                his_dic.pop(face_id, None)
        else:
            this_mtime = os.stat(track_id_dir).st_mtime
        modify_image_id(new_track_id_dir, track_id, time_stamp=this_mtime)
    PickleUtils.save_pickle('/home/manho/data/his_dic_remain.pkl', his_dic)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src', help='tracking folder', default=None)
    parser.add_argument(
        '-d', '--dst', help='new tracking folder', default='new_tracking')
    parser.add_argument(
        '-hd',
        '--history_dictionary',
        help='just extract his dictionary',
        action='store_true')
    args = parser.parse_args()
    if args.history_dictionary:
        get_history_in_send_folder()
    else:
        main(args.src, args.dst)
