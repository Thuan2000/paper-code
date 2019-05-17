'''
Cam worker for The Coffee House project

Created by @man
Last edit: Jan 10th, 2018
'''
import argparse
import time
from frame_reader import URLFrameReader, RabbitFrameReader
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from matcher import KdTreeMatcher
from tf_graph import FaceGraph
from config import Config
from rabbitmq import RabbitMQ
from tracker import TrackersList
from preprocess import Preprocessor
from cv_utils import FaceAngleUtils, CropperUtils


# TODO (@man): move this file to folder tch
def cam_worker_function(cam_url, area):
    '''
    Cam worker function
    '''
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))

    # Modify Config
    Config.Track.TRACKING_QUEUE_CAM_TO_CENTRAL = True

    rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                         (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))

    frame_counter = 0

    # Variables holding the correlation trackers and the name per faceid
    list_of_trackers = TrackersList()

    face_rec_graph = FaceGraph()
    face_extractor = FacenetExtractor(face_rec_graph)
    detector = MTCNNDetector(face_rec_graph)
    preprocessor = Preprocessor()
    matcher = KdTreeMatcher()
    if Config.CALC_FPS:
        start_time = time.time()
    if args.cam_url is not None:
        frame_reader = URLFrameReader(args.cam_url, scale_factor=1.5)
    else:
        frame_reader = RabbitFrameReader(rabbit_mq)

    try:
        while True:  # frame_reader.has_next():
            frame = frame_reader.next_frame()
            if frame is None:
                print("Waiting for the new image")
                list_of_trackers.check_delete_trackers(
                    matcher, rabbit_mq, history_mode=False)
                continue

            print("Frame ID: %d" % frame_counter)

            if Config.CALC_FPS:
                fps_counter = time.time()

            list_of_trackers.update_dlib_trackers(frame)

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                origin_bbs, points = detector.detect_face(frame)
                for i, origin_bb in enumerate(origin_bbs):
                    display_face, _ = CropperUtils.crop_display_face(
                        frame, origin_bb)
                    print("Display face shape")
                    print(display_face.shape)
                    if 0 in display_face.shape:
                        continue
                    cropped_face = CropperUtils.crop_face(frame, origin_bb)

                    # Calculate embedding
                    preprocessed_image = preprocessor.process(cropped_face)
                    emb_array, coeff = face_extractor.extract_features(
                        preprocessed_image)

                    # Calculate angle
                    angle = FaceAngleUtils.calc_angle(points[:, i])

                    # TODO: refractor matching_detected_face_with_trackers
                    matched_fid = list_of_trackers.matching_face_with_trackers(
                        frame, origin_bb, emb_array)

                    # Update list_of_trackers
                    list_of_trackers.update_trackers_list(
                        matched_fid, origin_bb, display_face, emb_array, angle,
                        area, frame_counter, matcher, rabbit_mq)

                    if Config.Track.TRACKING_QUEUE_CAM_TO_CENTRAL:
                        track_tuple = (matched_fid, display_face, emb_array,
                                       area, time.time(), origin_bb, angle)
                        rabbit_mq.send_tracking(
                            track_tuple,
                            rabbit_mq.RECEIVE_CAM_WORKER_TRACKING_QUEUE)

            # Check detete current trackers time
            list_of_trackers.check_delete_trackers(
                matcher, rabbit_mq, history_mode=False)

            frame_counter += 1
            if Config.CALC_FPS:
                print("FPS: %f" % (1 / (time.time() - fps_counter)))

    except KeyboardInterrupt:
        print('Keyboard Interrupt !!! Release All !!!')
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format(
                (frame_counter + 1) / (time.time() - start_time)))
        frame_reader.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'For demo only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--cam_url', help='your camera ip address', default=None)
    parser.add_argument(
        '-a',
        '--area',
        help='The area that your ip camera is recording at',
        default='None')

    args = parser.parse_args()

    # Run
    cam_worker_function(args.cam_url, args.area)
