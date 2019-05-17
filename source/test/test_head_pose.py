import cv2
# import mxnet as mx
import argparse
import time
from frame_reader import URLFrameReader

# from HeadPoseEstimator import HeadPoseEstimator
# from tf_graph import FaceGraph
# from face_detector import MTCNNDetector
# from cv_utils import FaceAngleUtils, CropperUtils


def test_function(cam_url):
    # face_rec_graph = FaceGraph()
    # detector = MTCNNDetector(face_rec_graph)
    # estimator = HeadPoseEstimator(model_prefix='../models/cpt', ctx=mx.cpu())
    if args.cam_url is not None:
        frame_reader = URLFrameReader(args.cam_url, scale_factor=1)
    else:
        return -1
    while True:
        frame = frame_reader.next_frame()
        if frame is None:
            print('Frame is None...')
            time.sleep(5)
            continue
        # origin_bbs, points = detector.detect_face(frame)
        # for i, origin_bb in enumerate(origin_bbs):
        #     cropped_face = CropperUtils.crop_face(frame, origin_bb)
        #     yaw = FaceAngleUtils.calc_angle(points[:, i])
        #     pitch = FaceAngleUtils.calc_face_pitch(points[:, i])
        #     print(cropped_face.shape)
        #     # resize_face = np.resize(cropped_face,(64, 64, 3))
        #     frame = FaceAngleUtils.plot_points(frame, points[:, i])
        #     # print('pitch-yaw angle of test1: {}'.format(estimator.predict(resize_face)))
        #     print('pitch-yaw angle: {}, {}'.format(pitch, yaw))
        #  # print('pitch-yaw angle: {}'.format(estimator.crop_and_predict(frame, [points[:, i]])))
        cv2.imshow('img', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--cam_url', help='your camera ip address', default=None)
    args = parser.parse_args()
    test_function(args.cam_url)
