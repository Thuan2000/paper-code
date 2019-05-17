import glob
from scipy import misc
from cv_utils import CropperUtils
from preprocess import Preprocessor, align_and_crop
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from matcher import FaissMatcher
from tracker_manager import TrackerManager
from tracking_utils import FaceInfo, Tracker
from config import Config
from face_detector import MTCNNDetector
from face_align import AlignCustom


def extract_embs(tracking_folder, preprocessor, face_extractor, detector):
    img_dirs = glob.glob(tracking_folder + '/*.jpg')
    embs = []
    labels = []
    aligner = AlignCustom()
    undetectable_faces = 0
    for img_dir in img_dirs:
        image_id = img_dir.split('/')[-1].replace('.jpg', '')
        splitted_image_id = image_id.split('_')
        bbox = splitted_image_id[-4:len(splitted_image_id)]
        bbox = '_'.join(bbox)
        origin_bb = splitted_image_id[1:5]
        origin_bb = [int(bbnum) for bbnum in origin_bb]
        time_stamp = float(splitted_image_id[5])
        img = misc.imread(img_dir)
        if detector is not None:
            origin_bb, points = detector.detect_face(img)
            print('nof detected faces ' + str(len(origin_bb)))
            if len(points) == 0 or len(origin_bb) != 1:
                undetectable_faces += 1
                print(undetectable_faces)
                continue
            else:
                preprocessed_image = preprocessor.process(
                    img, points[:, 0], aligner, 160)
        else:
            cropped_face = CropperUtils.reverse_display_face(img, bbox)
            preprocessed_image = preprocessor.process(cropped_face)

        # Extract feature
        emb_array, _ = face_extractor.extract_features(preprocessed_image)

        # For Matcher
        embs.append(emb_array)
        labels.append(image_id)
    return embs, labels


def main(matcher_path, test_path):
    m_trackers_paths = glob.glob(matcher_path + '/*')
    t_trackers_paths = glob.glob(test_path + '/*')
    tracker_manager = TrackerManager('test')
    matcher = FaissMatcher()
    preprocessor = Preprocessor()
    align_preprocessor = Preprocessor(algs=align_and_crop)
    face_rec_graph_face = FaceGraph()
    face_extractor = FacenetExtractor(
        face_rec_graph_face, model_path=Config.FACENET_DIR)
    detector = MTCNNDetector(face_rec_graph_face)

    # create matcher
    print('Creating matcher ...')
    for m_dir in m_trackers_paths:
        print('Processing ' + m_dir)
        face_id = m_dir.split('/')[-1]
        embs, labels = extract_embs(m_dir, preprocessor, face_extractor, None)
        face_id_labels = [face_id for i in range(len(labels))]
        matcher.update(embs, face_id_labels)

    # create tracker
    print('Creating trackers')
    for t_dir in t_trackers_paths:
        print('Processing ' + t_dir)
        embs, _ = extract_embs(t_dir, preprocessor, face_extractor, None)
        track_id = int(t_dir.split('/')[-1])

        first_emb = embs.pop()
        face_info = FaceInfo(None, first_emb, None, None, None, None)
        tracker_manager.current_trackers[track_id] = Tracker(
            track_id, face_info, None)
        for emb in embs:
            face_info = FaceInfo(None, emb, None, None, None, None)
            tracker_manager.current_trackers[track_id].update(face_info, None)
        len(tracker_manager.current_trackers)

    # test matching
    print('Test matching ...')
    for fid in tracker_manager.current_trackers:
        print('Processing: ' + str(fid))
        tops = tracker_manager.recognize_current_tracker(fid, matcher, None)
        print('Track_id {}, recognize: {}'.format(fid, tops))


main('/home/manho/data/testvin/matcher', '/home/manho/data/testvin/test')
