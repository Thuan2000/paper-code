import os
import sys
import glob
import pickle
import argparse
import numpy as np
from scipy import misc
from config import Config
from preprocess import default_preprocess
from tf_graph import FaceGraph
from face_extractor import FacenetExtractor
from cv_utils import CropperUtils


def extract_embs(model_path, data, margin, save_name,
                 nrof_samples=float('Inf')):
    save_name = os.path.expanduser(save_name.rstrip('/'))
    if not os.path.exists(os.path.dirname(save_name)):
        print('please enter a valid save_name')

    data = os.path.expanduser(data.rstrip('/'))
    id_dirs = glob.glob(os.path.join(data, '*'))

    id_dict = {}
    for id_dir in id_dirs:
        id_label = os.path.basename(id_dir)
        image_paths = glob.glob(os.path.join(id_dir, '*.*'))
        use_samples = min(nrof_samples, len(image_paths))
        for path in image_paths[:use_samples]:
            id_dict[path] = id_label

    tf = FaceGraph()
    extractor = FacenetExtractor(tf, model_path=model_path)
    nrof_imgs = 0
    emb_array = np.zeros((len(id_dict), 128))
    label_list = []
    image_ids = []
    print('reading images')
    for path, label in id_dict.items():
        try:
            img = misc.imread(path)
            coor_list = os.path.splitext(
                os.path.basename(path))[0].split('_')[-4:]
            bbox = np.array((coor_list), dtype=int)
            face_img = CropperUtils.crop_face(img, bbox, margin)
            face_img = default_preprocess(face_img)
            emb, coeff = extractor.extract_features(face_img).squeeze()
            emb_array[nrof_imgs, :] = emb
        except (ValueError, OSError) as e:
            print(e)
            print('skipping', path)
            continue
        nrof_imgs += 1
        label_list.append(label)
        image_ids.append(os.path.basename(path))
    emb_array = emb_array[:nrof_imgs]
    print('extracted {} images'.format(nrof_imgs))

    save = {'embs': emb_array, 'labels': label_list, 'image_ids': image_ids}

    save_file = os.path.join(save_name)
    print('result saved at', save_file)
    with open(save_file, 'wb') as f:
        pickle.dump(save, f)


def main(args):
    extract_embs(args.model_path, args.data, args.margin, args.save_name)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        '', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_path',
        help='path to the feature extractor',
        default=Config.FACENET_DIR)
    parser.add_argument(
        '--data', help='contruct as lfw format data_dir->ids->imgs')
    parser.add_argument(
        '--margin',
        help='expand pixels from original bbox',
        default=Config.Align.MARGIN,
        type=int)
    parser.add_argument(
        '--save_name', help='dir to save the pickle file', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
