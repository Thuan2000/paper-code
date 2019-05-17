import cv2
import os
from face_detector import MTCNNDetector
from tf_graph import FaceGraph
import numpy as np
import argparse
from frame_reader import URLFrameReader

# Config
OPENCV_DIR = '../models/opencv/'
OPENCV_FACE = 'haarcascade_frontalface_default.xml'
OPENCV_EYE = 'haarcascade_eye.xml'
OPENCV_GLASS_EYE = 'haarcascade_eye_tree_eyeglasses.xml'
OPENCV_LEFT_EYE = 'haarcascade_mcs_lefteye.xml'
OPENCV_RIGHT_EYE = 'haarcascade_mcs_righteye.xml'
OPENCV_MOUTH = 'haarcascade_mouth.xml'
SAMPLE_IMAGE = '../data/occlusion/Image/noocclusion/noocclusion_2.jpg'
SAMPLE_VIDEO = '../data/occlusion/Video/mask1.webm'
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
NO_OCCLUSION = 0
NO_MOUTH = 1
NO_EYES = 2
NO_FACE = 3
MULTI_FACE = 4


# Utils
def get_face_bound(face, gray, color):
    '''
    extract face region from frame
    :param face: face is tuple(x, y, w, h)
    :param gray: gray frame
    :param color: color frame
    :return:
    '''
    x, y, w, h = face
    face_gray = gray[y:y + h, x:x + w]
    face_color = color[y:y + h, x:x + w]
    return face_gray, face_color


def draw_box(image, shape, color, width=2):
    """
    Draw bouding box on image
    :param image:
    :param shape: shape of bouding box
    :param color:
    :param width:
    :return:
    """
    x, y, w, h = shape
    cv2.rectangle(image, (x, y), (x + w, y + h), color, width)


def draw_eye(image, point, width, height):
    x, y = point
    i_x = x - int(width / 2)
    i_y = y - int(height / 2)
    i_w = width
    i_h = height
    return (i_x, i_y, i_w, i_h)


def draw_mouth(image, right, left):
    m_h = left[0] - right[0]
    m_w = m_h + int(m_h / 2)
    m_x = right[0] - int(m_h / 4)
    m_y = right[1] - int(m_h / 2)
    return (m_x, m_y, m_w, m_h)


def put_text_on_image(img, content, color, location="bottom"):
    """
    Draw text on img
    :param img:
    :param content:
    :param color:
    :param location:
    :return:
    """
    h, w = img.shape[:2]
    if location == "bottom":
        text_location = (20, h - 10)
    else:
        text_location = (20, 100)
    cv2.putText(img, content, text_location, cv2.FONT_HERSHEY_SIMPLEX, 2, color,
                2, cv2.LINE_AA)


def display_result_board(board, result):
    font = cv2.FONT_HERSHEY_SIMPLEX

    def status(x):
        if x == 1:
            return 'DETECTED'
        else:
            return 'MISSING'

    face_status = "Face : {}".format(status(result['face']))
    leye_status = "Left Eye : {}".format(status(result['left_eye']))
    reye_status = "Right Eye : {}".format(status(result['right_eye']))
    mouth_status = "Mouth : {}".format(status(result['mouth']))
    cv2.putText(board, face_status, (50, 50), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(board, leye_status, (50, 75), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(board, reye_status, (50, 100), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(board, mouth_status, (50, 125), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)


def process_result(result):
    if result['face'] == 0:
        return NO_FACE
    if result['right_eye'] == 0 or result['left_eye'] == 0:
        return NO_EYES
    if result['mouth'] == 0:
        return NO_MOUTH
    return NO_OCCLUSION


def show_information(frame, current_frame, frames_per_state, state_validation):
    state_color = GREEN if state_validation else RED
    state_text = "Accept" if state_validation else "Reject"
    frame_h, frame_w = frame.shape[:2]
    fps = "{0}/{1}".format(current_frame, frames_per_state)
    put_text_on_image(frame, fps, BLUE, "top")
    draw_box(frame, (0, 0, frame_w, frame_h), state_color)
    put_text_on_image(frame, state_text, state_color)


# Detection
class CVDetection(object):
    '''
    Use opencv cascade to detect object
    '''
    scaleFactor = 1.5
    minNeighbors = 5
    cascade_name = OPENCV_FACE

    def __init__(self, model_path):
        """
        Load opencv cascade model
        :param model_path: path to cascade model
        """
        if not os.path.exists(model_path):
            raise Exception("Model path invalid")
        self.detector = cv2.CascadeClassifier(
            os.path.join(model_path, self.cascade_name))

    def detect(self, image):
        return self.detector.detectMultiScale(image, self.scaleFactor,
                                              self.minNeighbors)


class MTCNNFaceDetection(object):
    '''
    Use mtcnn model to detect face in frame
    '''

    def __init__(self):
        face_graph = FaceGraph()
        self.detector = MTCNNDetector(face_graph)

    def detect(self, image, **kwargs):
        faces = []
        margin = 0
        img_h, img_w = image.shape[:2]
        origin_bbs, landmarks = self.detector.detect_face(image, **kwargs)

        landmarks = landmarks.reshape(-1, 10).tolist()
        for origin_bb, landmark in zip(origin_bbs, landmarks):
            face = {}
            det = np.squeeze(origin_bb)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_w)
            bb[3] = np.minimum(det[3] + margin / 2, img_h)
            x = bb[0]
            y = bb[1]
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            face['face'] = (x, y, w, h)

            points = []
            # 0: right eye
            # 1: left eye
            # 2: nose
            # 3: right of mount
            # 4: left of mount
            for ix, iy in zip(landmark[:5], landmark[5:]):
                points.append((ix, iy))
            # Draw right eye
            eye_width = int(w / 3)
            right_eye = draw_eye(image, points[0], eye_width, eye_width)
            left_eye = draw_eye(image, points[1], eye_width, eye_width)
            # Draw mouth
            mouth = draw_mouth(image, points[3], points[4])

            face['right_eye'] = right_eye
            face['left_eye'] = left_eye
            face['mouth'] = mouth
            faces.append(face)
        return faces


class FaceDetection(object):
    '''
    Use face detection models to detect face in frame
    '''

    def __init__(self, model_path=OPENCV_DIR, model='opencv'):
        '''
        :param model_dir: model is stored in here
        :param model: model name, e.g: opencv, mtcnn
        '''
        if model == 'opencv':
            self.detector = CVFaceDetection(model_path)
        else:
            self.detector = MTCNNFaceDetection()

    def detect(self, image):
        return self.detector.detect(image)


class CVFaceDetection(CVDetection):
    scaleFactor = 1.1
    minNeighbors = 3
    cascade_name = OPENCV_FACE


class CVEyesDetection(CVDetection):
    scaleFactor = 1.1
    minNeighbors = 3
    cascade_name = OPENCV_EYE


class CVLeftEyesDetection(CVDetection):
    scaleFactor = 1.1
    minNeighbors = 3
    cascade_name = OPENCV_LEFT_EYE


class CVRightEyesDetection(CVDetection):
    scaleFactor = 1.1
    minNeighbors = 3
    cascade_name = OPENCV_RIGHT_EYE


class CVMouthDetection(CVDetection):
    scaleFactor = 1.1
    minNeighbors = 1
    cascade_name = OPENCV_MOUTH


class OcclusionDetection(object):

    def __init__(self, model_dir=OPENCV_DIR):
        self.face_detector = FaceDetection(model='mtcnn', model_path=model_dir)
        self.eye_detector = CVEyesDetection(model_dir)
        self.mouth_detector = CVMouthDetection(model_dir)
        self.reye_detector = CVRightEyesDetection(model_dir)
        self.leye_detector = CVLeftEyesDetection(model_dir)

    def detect(self, img):
        result = {
            'face': 0,
            'right_eye': 0,
            'left_eye': 0,
            'mouth': 0,
            'multiface': 0,
        }
        bgr_frame = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_detector.detect(img)
        if len(faces) == 0:
            return result
        elif len(faces) > 1:
            result['multiface'] = 1
            return result
        else:
            face = faces[0]['face']
            right_eye = faces[0]['right_eye']
            left_eye = faces[0]['left_eye']
            mouth = faces[0]['mouth']
            draw_box(img, faces[0]['right_eye'], GREEN, 1)
            draw_box(img, faces[0]['left_eye'], GREEN, 1)
            draw_box(img, faces[0]['mouth'], GREEN, 1)
            draw_box(img, face, RED)
            result['face'] = 1
            # Check right eye existence
            roi_reye_gray, roi_reye_color = get_face_bound(right_eye, gray, img)
            reyes = self.reye_detector.detect(roi_reye_gray)
            if len(reyes) > 0:
                result['right_eye'] = 1
            for reye in reyes:
                draw_box(roi_reye_color, reye, (255, 221, 0))
            # Check left eye existence
            roi_leye_gray, roi_leye_color = get_face_bound(left_eye, gray, img)
            leyes = self.leye_detector.detect(roi_leye_gray)
            if len(leyes) > 0:
                result['left_eye'] = 1
            for leye in leyes:
                draw_box(roi_leye_color, leye, (255, 221, 0))

            # Check mouth existence
            roi_mouth_gray, roi_mouth_color = get_face_bound(mouth, gray, img)
            mouths = self.mouth_detector.detect(roi_mouth_gray)
            if len(mouths) > 0:
                result['mouth'] = 1
            for mouth in mouths:
                draw_box(roi_mouth_color, mouth, BLUE)
        return result


def occlusion_dection_image(image_path, detector):
    # load image
    img = cv2.imread(image_path)
    # detect occlusion
    detected_result = detector.detect(img)
    image_label = process_result(detected_result)
    state_validation = True if image_label == NO_OCCLUSION else False

    state_color = GREEN if state_validation else RED
    state_text = "Accept" if state_validation else "Reject"
    put_text_on_image(img, state_text, state_color)
    cv2.imshow('Result', img)
    cv2.waitKey(0)


def occlusion_dection_video(video_path, detector):
    frame_reader = URLFrameReader(video_path, scale_factor=1)
    frames_per_state = 4
    state_correct = 0
    curent_frame = 0
    # Opening phase
    try:
        for i in range(frames_per_state):
            curent_frame += 1
            frame = frame_reader.next_frame()
            detected_result = detector.detect(frame)
            frame_label = process_result(detected_result)
            if frame_label == NO_OCCLUSION:
                state_correct += 1
            # fps = "{0}/{1}".format(curent_frame, frames_per_state)
            # put_text_on_image(frame, fps, BLUE, "top")
            # cv2.imshow('frame', frame)

        state_validation = True if state_correct >= 1 else False
        state_correct = 0
        curent_frame = 0

        # Realtime phase
        while frame_reader.has_next():
            result_board = 255 * np.ones((400, 400, 3))
            frame = frame_reader.next_frame()
            curent_frame += 1
            show_information(frame, curent_frame, frames_per_state,
                             state_validation)
            detected_result = detector.detect(frame)
            frame_label = process_result(detected_result)
            if frame_label == NO_OCCLUSION:
                state_correct += 1

            if curent_frame >= frames_per_state:
                state_validation = True if state_correct >= 1 else False
                curent_frame = 0
                state_correct = 0
            display_result_board(result_board, detected_result)
            cv2.imshow('frame', frame)
            cv2.imshow('result', result_board)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest='image_path', default=None)
    parser.add_argument('--v', dest='video_path', default=0)
    parser.add_argument('--m', dest='model_dir', default=OPENCV_DIR)
    args = parser.parse_args()
    image_path = args.image_path
    video_path = args.video_path
    model_dir = args.model_dir
    occlusion_detection = OcclusionDetection(model_dir)
    if image_path:
        occlusion_dection_image(image_path, occlusion_detection)
    else:
        occlusion_dection_video(video_path, occlusion_detection)
