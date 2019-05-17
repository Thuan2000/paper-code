from config import Config
import numpy as np
import cv2
import base64
import os
import pickle
import scipy
import glob
from operator import itemgetter
from scipy import misc
import portalocker as locker
import shutil
import functools
import zlib
from math import atan, degrees
import re
from statistics import mode
import imageio
import io
from scipy.spatial.distance import cdist


def get_area(rect):
    """
    Return bounding_box's area given its position
    >>> get_area([0, 1, 2, 3])
    4
    """
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def get_area_list(rect_list):
    """
    Return bounding_boxes' list given their position
    >>> get_area_list([[0,1,2,3],[1,3,5,7]])
    array([ 4, 16])
    """
    area_list = []
    for rect in rect_list:
        area_list.append(get_area(rect))
    return np.array(area_list)

def padded_landmark(landmarks, bouding_box, padded_bb_str, margin):
    padding_distance = int(margin/2)
    #convert landmark from 1x10 to 2x5
    landmarks = landmarks.reshape(2,-1)
    landmark_in_BB = landmarks - bouding_box[:2].reshape(-1,1)
    x1,y1,x2,y2 = [int(i) for i in padded_bb_str.split('_')]
    landmark_in_padded = landmark_in_BB  + np.array([[x1+padding_distance],[y1+padding_distance]])
    return landmark_in_padded.flatten()


class CropperUtils:
    """
    Utility for cropping face
    >>> edit_image = CropperUtils()
    TODO: May extract this class, each face is an instance cotain frame and bb,
    TODO: has func like crop_face, reverse?
    """

    @staticmethod
    def crop_face(frame,
                  bounding_box,
                  margin=Config.Align.MARGIN,
                  return_size=Config.Align.IMAGE_SIZE):
        """
        :param frame: Image frame
        :param bounding_box: bounding box for face in frame
        :return: resize image of size 160x160

        >>> import numpy; \
            numpy.random.seed(0); \
            temp = numpy.random.randn(500, 500, 3); \
            CropperUtils.crop_face(temp, [100, 100, 200, 200]).shape
        (160, 160, 3)
        """
        h, w, _ = frame.shape
        det = np.squeeze(bounding_box)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, w)
        bb[3] = np.minimum(det[3] + margin / 2, h)
        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
        resized = misc.imresize(
            cropped, (return_size, return_size), interp='bilinear')
        return resized

    @staticmethod
    def crop_display_face(frame,
                          bounding_box,
                          margin=Config.Align.MARGIN,
                          roi_rate_w=Config.Align.ROI_RATE_W,
                          aspect_ratio=Config.Align.ASPECT_RATIO,
                          upper_rate=Config.Align.UPPER_RATE,
                          lower_rate=Config.Align.LOWER_RATE):
        """
            Return:
            - crop face for display with aspect ratio w/h = 2/3
            - origin bb in according to display face
        >>> import numpy; \
            numpy.random.seed(0); \
            temp = numpy.random.randn(500, 500, 3); \
            image, bounding_box = CropperUtils.crop_display_face(temp, [100, 100, 200, 200]); \
            print(str(image.shape) + " " + str(bounding_box))
        (313, 210, 3) 55_75_-55_-138
        """
        # preprocess bb before crop display face
        # TODO: Extract this outside
        h, w, _ = frame.shape
        det = np.squeeze(bounding_box)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, w)
        bb[3] = np.minimum(det[3] + margin / 2, h)

        h, w, _ = frame.shape
        mid_y = (bb[1] + bb[3]) // 2
        mid_x = (bb[0] + bb[2]) // 2
        crop_w = bb[2] - bb[0]

        half_old_w = crop_w / 2
        half_new_w = int((half_old_w * (100 + roi_rate_w) / 100))
        half_new_h = int(half_new_w * aspect_ratio)

        from_y = mid_y - int(half_new_h * upper_rate)
        to_y = mid_y + int(half_new_h * lower_rate)
        from_x = mid_x - half_new_w
        to_x = mid_x + half_new_w

        pad_left = abs(min(0, from_x))
        pad_right = abs(min(0, w - to_x))
        pad_top = abs(min(0, from_y))
        pad_bottom = abs(min(0, h - to_y))
        # print('Pad lef, righ, top, bottom', pad_left, pad_right, pad_top, pad_bottom)

        if pad_top > 0:
            from_y += pad_top
            to_y += pad_top
            pad_bottom = abs(min(0, h - to_y))
            pad_top = 0

        padded = np.pad(
            frame, ((int(pad_top), int(pad_bottom)),
                    (int(pad_left), int(pad_right)), (0, 0)),
            mode='edge')

        # update coord according to pad
        from_x += pad_left
        to_x += pad_right + pad_left
        from_y += pad_top

        display_face = padded[int(from_y):int(to_y), int(from_x):int(to_x), :]
        h_display, w_display, _ = display_face.shape
        padped_bb = np.zeros(4, dtype=np.float)
        padped_bb[0] = bb[0] - from_x + pad_left
        padped_bb[1] = bb[1] - from_y
        if padped_bb[0] < 0:
            padped_bb[0] = w_display - padped_bb[0]
        if padped_bb[1] < 0:
            padped_bb[1] = h_display - padped_bb[1]
        padped_bb[2] = (bb[2] - bb[0]) + padped_bb[0]
        padped_bb[3] = (bb[3] - bb[1]) + padped_bb[1]

        padded_bb_str = '%d_%d_%d_%d' % (padped_bb[0], padped_bb[1],
                                         padped_bb[2], padped_bb[3])

        return display_face, padded_bb_str

    @staticmethod
    def reverse_display_face(display_face, padded_bb_str):
        """
        return cropped: image for this display face
        >>> import numpy; \
            numpy.random.seed(0); \
            temp = numpy.random.randn(500, 500, 3); \
            CropperUtils.reverse_display_face(temp, "55_75_-55_-138").shape
        BB [55.0, 75.0, -55.0, -138.0]
        (160, 160, 3)
        """

        bb = padded_bb_str.split('_')
        bb = list(map(float, bb))
        cropped = CropperUtils.crop_face(display_face, bb)
        return cropped


#     @staticmethod
#     def crop_faces(frame, bbs):
#         return [CropperUtils.crop_face(frame, bb) for bb in bbs]

#     @staticmethod
#     def crop_display_faces(frame, bbs):
#         return [CropperUtils.crop_display_face(frame, bb) for bb in bbs]

def is_inner_of_range(bb, frame_shape):
    """
    >>> is_inner_of_range([100, 100, 150, 150], [160, 160, 3])
    False
    >>> is_inner_of_range([-1, 100, 150, 150], [160, 160, 3])
    face is inner of range!
    True
    >>> is_inner_of_range([100, -1, 150, 150], [160, 160, 3])
    face is inner of range!
    True
    >>> is_inner_of_range([100, 100, 161, 150],[160, 160, 3])
    face is inner of range!
    True
    >>> is_inner_of_range([100, 100, 150, 161], [160, 160, 3])
    face is inner of range!
    True
    """
    h, w, _ = frame_shape
    if bb[0] < 0 or bb[2] > w or bb[1] < 0 or bb[3] > h:
        print('face is inner of range!')
        return True
    return False


def mkdir_on_check(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.mkdir(dir_)


class PickleUtils:

    @staticmethod
    def save_pickle(pickle_path, value):
        """
        Save value to pickle file
        >>> PickleUtils().save_pickle(r'test/doctest_sample.pkl', "Sample Pickle")

        """
        file_pickle = open(pickle_path, 'wb')
        locker.lock(file_pickle, locker.LOCK_EX)
        pickle.dump(value, file_pickle)
        file_pickle.close()

    @staticmethod
    def read_pickle(pickle_path, default=None):
        """
        Read a pickle file
        Test along with save_pickle
        >>> PickleUtils().read_pickle(r'test/doctest_sample.pkl', [])
        'Sample Pickle'
        >>> PickleUtils().read_pickle(r'test/doctest_sample_non_exist.pkl', [])
        []
        """
        if os.path.exists(pickle_path):
            file_pickle = open(pickle_path, 'rb')
            locker.lock(file_pickle, locker.LOCK_EX)
            return_value = pickle.load(file_pickle)
            file_pickle.close()
            return return_value
        else:
            return default


def atuan_calc_distance_many_to_many_infolder(folder_path):
    imgs = {}
    valid_images = [".jpg"]
    folder_name = folder_path.split('/')[-1]
    live_dir = '/media/minhmanho/42B6761CB67610A1/1st-EyeQ/Data/vin-data/qa/qa-database/live'
    for filename in os.listdir(folder_path):
        print(filename)
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_images:
            continue
        image_id = os.path.splitext(filename)[0].split('|')[1]
        imgs[filename] = pickle.load(
            open(os.path.join(live_dir, image_id + '.pkl'), 'rb'))[1]

    matching_matrix = []
    for filename in imgs:
        matching_matrix.append([
            np.sum(np.square(np.subtract(imgs[fid], imgs[filename])), 0)
            for fid in imgs
        ])

    print('Save matching_matrix of the folder {} as csv... Done'.format(
        folder_name))
    np.savetxt('{}.csv'.format(folder_name), matching_matrix)

    return matching_matrix


def show_frame(frame, name='Frame', bb=None, pt=None, id=None, wait_time=1):
    """
    >>> import cv2; \
        image = cv2.imread("../data/matching/set1/0_0.jpg"); \
        show_frame(image, wait_time = 1)
    True
    """

    if Config.Mode.SHOW_FRAME:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if bb is not None:
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 165, 255),
                          2)
        if id is not None:
            cv2.putText(frame, id, (int(bb[0]), int(bb[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if pt is not None:
            pt_x = pt[0:5]
            pt_y = pt[5:10]
            for cx, cy in zip(pt_x, pt_y):
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow(name, frame)
        k = cv2.waitKey(100)
    return True


def encode_image(image):
    """
        Input: Original Image (type: ndarray)
        Output: Encoded Image (type: string)
    >>> import cv2; \
        image = cv2.imread("../data/matching/set1/0_0.jpg"); \
        encoded_string = encode_image(image); \
        print(len(encoded_string)); \
        decoded_image = decode_image(encoded_string); \
        print((image == decoded_image).all())
    25892
    True
    """
    cvt_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', cvt_img)
    byte_ = base64.b64encode(buffer).decode('ascii')
    # compressed = zlib.compress(byte)
    # ascii_str = compressed.decode('ascii')
    # print('ascc', type(ascii_str))
    return byte_


def decode_image(_byte):
    """
        Input: Encoded Image (type : string)
        Output: Original Image (type: ndarray)

        Doctest for this function was corvered by function encoded_image above
        >>> print("Tested along with encoded_image")
        Tested along with encoded_image
    """
    # byte = ascii_str.encode("ascii")
    # decompressed = zlib.decompress(compressed)
    _ascii = _byte.encode("ascii")
    _base64 = base64.b64decode(_ascii)
    _img = cv2.imdecode(np.fromstring(_base64, dtype=np.uint8), 1)
    img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    return img


def decode_ndarray(array_str):
    """
    >>> A = "[1 2 3 4]"; \
        decode_ndarray(A)
    array([ 1.,  2.,  3.,  4.])
    """
    return np.array(array_str.strip('[').strip(']').split()).astype(np.float)


def encode_ndarray(ndarray):
    """
    >>> import numpy; \
        A = np.array([[0,1,2],[4,5,6]], dtype = 'f'); \
        encode_ndarray(A)
    '[ 0.  1.  2.] [ 4.  5.  6.]'
    """
    return ' '.join([str(feat) for feat in ndarray])


def find_id(emb, emb_list):
    """
    >>> import numpy; \
        emb = np.random.randn(1, 128); \
        emb_list = []; \
        emb_list.append(emb); \
        emb_list.append(np.random.randn(1,128)); \
        find_id(emb, emb_list)
    (0, 0.0)
    """
    min_dist = 20000
    nearest_id = 0
    for id_, embs in enumerate(emb_list):
        dists = np.sum(np.square(np.subtract(embs, emb)), 1)
        min_dist_of_this_id = np.min(dists)
        if min_dist_of_this_id < min_dist:
            min_dist = min_dist_of_this_id
            nearest_id = id_
    return nearest_id, min_dist


def run_rpca(tracker):
    """
    >>> run_rpca(_)
    'a low-rank tracker'
    """
    return 'a low-rank tracker'


def calc_min_min_distance(tracker, emb_reg_dict):
    print('Calculate min-min + a tuans distance')
    min_dict = {}
    for tracker_element in tracker:
        list_sub_embeddings = []
        print(tracker_element.embedding[0])
        print('------------')
        print(emb_reg_dict[list(emb_reg_dict.keys())[0]])
        list_sub_embeddings = [
            np.subtract(tracker_element.embedding[0], emb_reg_dict[fid])
            for fid in emb_reg_dict.keys()
        ]
        dists = np.sum(np.square(list_sub_embeddings), 1)
        min_dist = np.min(dists)
        min_dist_index = list(emb_reg_dict.keys())[np.argmin(dists)]
        min_dict[min_dist_index] = min_dist

    m_index = min(min_dict, key=min_dict.get)
    return m_index, min_dict[m_index]


def mean_normalize(X):
    """
    >>> A = np.array([[0, 1, 2],[4, 5, 6]], dtype = 'f'); \
        mean_normalize(A)
    array([[-2., -2., -2.],
           [ 2.,  2.,  2.]], dtype=float32)
    """
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X
    return X


def normalize_number_of_column(immatrix1, immatrix2):
    '''
    #TODO (@man): tu tu viet doc cho ham nay
    '''
    num_data1, _ = immatrix1.shape
    num_data2, _ = immatrix2.shape
    if num_data1 < 4:
        immatrix1 = np.concatenate((immatrix1, immatrix1, immatrix1, immatrix1),
                                   axis=0)
    if num_data2 < 4:
        immatrix2 = np.concatenate((immatrix2, immatrix2, immatrix2, immatrix2),
                                   axis=0)
    if num_data1 == num_data2:
        return immatrix1, immatrix2

    return immatrix1, immatrix2


def calc_msm_distance(tracker, reg_labels_emb_dict):
    '''
    #TODO (@man): tu tu viet doc cho ham nay
    '''
    print('Calculate Mutual Subspace Method distance')
    immatrix1 = np.array(
        [tracker_element.embedding[0] for tracker_element in tracker])

    max_dict = {}
    for fid in reg_labels_emb_dict.keys():
        print('Loop: {}'.format(fid))
        immatrix2 = np.array(reg_labels_emb_dict[fid])

        immatrix1 = mean_normalize(immatrix1)
        immatrix2 = mean_normalize(immatrix2)

        immatrix1, immatrix2 = normalize_number_of_column(immatrix1, immatrix2)
        kmin = min(min(immatrix1.shape), min(immatrix2.shape)) - 1
        print(immatrix1.shape)
        print('------------')
        print(immatrix2.shape)

        U1, S1, _ = scipy.sparse.linalg.svds(immatrix1, k=kmin)
        U2, S2, _ = scipy.sparse.linalg.svds(immatrix2, k=kmin)

        U1 = np.array(U1)
        U2 = np.array(U2)
        minn = min(U1.shape[0], U2.shape[0])
        if minn < 10:
            U1 = U1[0:minn, :]
            U2 = U2[0:minn, :]
        A = U1.transpose().dot(U2)
        print(U1)
        print(U2)
        print(A)
        _, cmsm, _ = scipy.sparse.linalg.svds(A, k=1)

        max_dict[fid] = cmsm
    print(max_dict)
    m_index = max(max_dict, key=max_dict.get)
    return m_index, max_dict[m_index]


def sample_pixel():
    return np.ones((1, 1, 3), dtype=np.uint8)


def sample_ndarray():
    return np.array([[1], [1]], dtype=float)


def sample_array():
    return np.array(([1]))


def check_overlap(bb1, bb2):
    """
    Check overlap between 2 bounding boxes
    Return: float in [0,1]
    >>> check_overlap([1, 1, 3, 4], [1, 2, 3, 5])
    0.5
    """
    # assert bb1[0] < bb1[2]
    # assert bb1[1] < bb1[3]
    # assert bb2[0] < bb2[2]
    # assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def is_inner_bb(big_bb, small_bb):
    '''
    Check that bb2 inner bb1?
    '''
    if (small_bb[0] < big_bb[0] or small_bb[2] > big_bb[2] or
            small_bb[1] < big_bb[1] or small_bb[3] > big_bb[3]):
        return False
    return True


def calc_bb_distance(bb1, bb2):
    '''
    calc using centerpoint
    '''
    return np.sqrt((
        (bb1[2] - bb1[0]) / 2 + bb1[0] - (bb2[2] - bb2[0]) / 2 + bb2[0])**2 + (
            (bb1[3] - bb1[1]) / 2 + bb1[1] - (bb2[3] - bb2[1]) / 2 + bb2[1])**2)


def wait_recognition_process(wait_function, recogntion_function, rabbit_mq,
                             matched_fid):
    """
    This function is created for waiting the wait_function
    """
    flag = True
    while flag:
        print('Recognizing ID {} ...'.format(matched_fid))
        flag = wait_function(recogntion_function, rabbit_mq, matched_fid)


def refresh_folder(path):
    '''
    come come
    '''
    if not is_exist(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def get_avg_dists(labels, dists):
    result = {}
    nof_labels = len(labels)
    for i, label in enumerate(labels):
        if not label in result:
            result[label] = []
        result[label].append(dists[i])
    for fid in result:
        nof_this_fid = len(result[fid])
        result[fid] = {
            'dist': sum(result[fid]) / len(result[fid]),
            'rate': nof_this_fid / nof_labels
        }
    return result

def get_biggest_folder_number(path):
    tracking_dirs = glob.glob(path + '/*')
    if tracking_dirs == []:
        number_of_existing_trackers = -1
    else:
        lof_int_trackid = []
        for tracking_dir in tracking_dirs:
            try:
                track_id = int(os.path.basename(tracking_dir))
                lof_int_trackid.append(track_id)
            except ValueError:
                pass
        number_of_existing_trackers = max(lof_int_trackid)
    return number_of_existing_trackers

def create_if_not_exist(path):
    '''
    come come
    '''
    if not is_exist(path):
        os.makedirs(path)


def is_exist(path):
    return os.path.exists(path)


def clear_folder(path):
    if is_exist(path):
        shutil.rmtree(path)


def clear_session_folder():
    refresh_folder(Config.Dir.SESSION_DIR)
    refresh_folder(Config.Dir.DB_DIR)
    refresh_folder(Config.Dir.LIVE_DIR)
    refresh_folder(Config.Dir.REG_DIR)


def get_img_url_by_id(str_dir, str_id):
    '''
    get all images urls that have str_id in its name
    '''
    return glob.glob(os.path.join(str_dir, '{}*.jpg'.format(str_id)))


def track_calls(func):

    @functools.wraps(func)
    def wrapper(*args, **kargs):
        wrapper.call_count += 1
        return func(*args, **kargs)

    wrapper.call_count = 0
    return wrapper


class DistanceUtils:

    @staticmethod
    def l2_distance(embs, emb):
        '''
        >>> a = [1, 2, 3, 4]; \
            b = [1, 1, 1, 1]; \
            DistanceUtils.l2_distance(a, b)
        3.7416573867739413
        >>> a = [[1, 2, 3, 4]]; \
            b = [1, 1, 1, 1]; \
            DistanceUtils.l2_distance(a, b)
        3.7416573867739413
        >>> a = [[1, 1, 1, 1], [1, 1, 1, 1]]; \
            b = [1, 1, 1, 1]; \
            DistanceUtils.l2_distance(a, b)
        array([ 0.,  0.])
        '''
        embs = np.squeeze(embs)
        if len(embs.shape) > 1:
            return np.sqrt(np.sum(np.square(np.subtract(embs, emb)), 1))
        else:
            return np.sqrt(np.sum(np.square(np.subtract(embs, emb))))

    @staticmethod
    def l1_distance(embs, emb):
        return np.abs(np.subtract(embs, emb))


def is_empty(a):
    '''
    >>> is_empty([])
    True
    >>> is_empty(None)
    True
    >>> is_empty({})
    True
    >>> is_empty([1])
    False
    >>> is_empty({1:2})
    False
    '''
    if a:
        return False
    return True


def compress(s):
    print('Before', len(s))
    compressed = zlib.compress(s)
    print('After', len(compressed))
    zlib.compress
    return compressed


def extract(s):
    return zlib.decompress(s)


def calc_bb_size(bounding_box):
    return abs(bounding_box[2] - bounding_box[0]) * abs(bounding_box[3] -
                                                        bounding_box[1])


def calc_bb_percentage(bounding_box, frame_shape):
    frame_bb = [0, 0, frame_shape[0], frame_shape[1]]
    return calc_bb_size(bounding_box) * 100 / calc_bb_size(frame_bb)


def draw_img(frame, bounding_boxes, name):
    bb0 = int(bounding_boxes[0])
    bb1 = int(bounding_boxes[1])
    bb2 = int(bounding_boxes[2])
    bb3 = int(bounding_boxes[3])
    cv2.rectangle(frame, (bb0, bb1), (bb2, bb3), (0, 165, 255), 2)

    cv2.putText(frame, str(name), (int(bb0 + (bb2 - bb0) / 2), bb1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def update_dict(dct, key, value):
    if key in dct:
        dct[key] += value
    else:
        dct[key] = value
    return dct


def print_out_frequency(labels_list, dists_list):
    labels_dict = {}
    dists_dict = {}
    write_line = []
    nrof_labels = len(labels_list)
    # print('nrof_labels ', nrof_labels)
    for i, label in enumerate(labels_list):
        labels_dict = update_dict(labels_dict, label, 1)
        dists_dict = update_dict(dists_dict, label, dists_list[i])
        # if dists_list[i] > Config.Track.HISTORY_RETRACK_THRESHOLD:
        #     labels_dict = update_dict(labels_dict, Config.Matcher.NEW_FACE, 1)
        #     dists_dict = update_dict(dists_dict, Config.Matcher.NEW_FACE, dists_list[i])

    sorted_labels_dict = sorted(
        labels_dict.items(), key=itemgetter(1), reverse=True)
    for i, (k, v) in enumerate(sorted_labels_dict):
        if i < 5:
            line = 'id: {}, fre: {}%, mean dist: {}'.format(
                k,
                round(labels_dict[k] / nrof_labels, 2) * 100,
                round(dists_dict[k] / labels_dict[k], 4))
            write_line.append(line)
        else:
            break
    rank1 = sorted_labels_dict[0][0]
    if labels_dict[rank1] / nrof_labels < Config.Track.HISTORY_RETRACK_MINRATE:
        return Config.Matcher.NEW_FACE, '\n'.join(write_line)
    return rank1, '\n'.join(write_line)


def calc_iou(bb_test, bb_gt):
    """
	Computes IUO between two bboxes in the form [x1,y1,x2,y2]
	"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])\
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return (o)


def find_most_common(ids):
    try:
        top = mode(ids)
    except:
        top = ids[0]
    return top


def base64str_to_frame(url):
        try:
            # print(url[:30])
            image_str = re.search(r'base64,(.*)', url).group(1)
            image_format = url[11:].split(';')[0]
        except Exception as e:
            print('Error parsing image from url', e)
            return False, None

        try:
            # print(image_format)
            image = imageio.imread(
                io.BytesIO(base64.b64decode(image_str)), image_format)
            image = image[:, :, :3]  # TODO: Convert RGBA to RGB
        except Exception as e:
            print('Error parsing image string', e)
            return False, None
        return True, image


def frame_to_base64str(frame):
    return 'TODO'

# TODO: replace hnswlib
def compute_nearest_neighbors_distances(embs_1, embs_2):
    if embs_1.size == 0 or embs_2.size == 0:
        return np.array([])
    distances_matrix = cdist(embs_1, embs_2)
    min_vector = distances_matrix.min(axis=1)
    return min_vector.flatten()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
