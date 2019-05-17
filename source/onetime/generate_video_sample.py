'''
Perform detection + tracking + recognition
Run: python3 generic_detection_tracking.py -c <camera_path> default <rabbit_mq>
                                                                    (for reading frames)
                                           -a <area> default 'None'
                                                                    (for area)
                                           -wi True default False
                                                                    (for writing all face-tracks)
                                           -vo True default False
                                                                    (for write tracking video)
'''
import argparse
from frame_reader import URLFrameReader, RabbitFrameReader
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from config import Config
from rabbitmq import RabbitMQ
from preprocess import Preprocessor
from cv_utils import CropperUtils, PickleUtils


class FrameSample:

    def __init__(self):
        self.frame_id = 0
        self.origin_bbs = []
        self.points = []
        self.read_image = []
        self.embs = []


rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))


def generate_video_sample(cam_url, area):
    '''generating'''
    print('Generating... ')
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))
    # Variables for tracking faces
    frame_counter = 0

    # Variables holding the correlation trackers and the name per faceid
    frame_sample = {}

    face_rec_graph = FaceGraph()
    face_extractor = FacenetExtractor(face_rec_graph)
    detector = MTCNNDetector(face_rec_graph)
    preprocessor = Preprocessor()
    if args.cam_url is not None:
        frame_reader = URLFrameReader(args.cam_url, scale_factor=1)
    else:
        frame_reader = RabbitFrameReader(rabbit_mq)

    try:
        while True:  # frame_reader.has_next():
            frame = frame_reader.next_frame()
            frame_sample[frame_counter] = FrameSample()
            frame_sample[frame_counter].read_image = frame
            if frame is None:
                print("Waiting for the new image")
                continue

            print("Frame ID: %d" % frame_counter)

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                origin_bbs, points = detector.detect_face(frame)
                frame_sample[frame_counter].origin_bbs = origin_bbs
                frame_sample[frame_counter].points = points
                for _, origin_bb in enumerate(origin_bbs):
                    cropped_face = CropperUtils.crop_face(frame, origin_bb)

                    # Calculate embedding
                    preprocessed_image = preprocessor.process(cropped_face)
                    emb_array, coeff = face_extractor.extract_features(
                        preprocessed_image)
                    frame_sample[frame_counter].embs.append(emb_array)

            frame_counter += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt !!! Release All !!!')
        print('Saved this video sample as ../session/db/sample.pkl')
        PickleUtils.save_pickle('../session/db/sample.pkl', frame_sample)


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
    parser.add_argument(
        '-wi',
        '--write_images',
        help='Write all face-tracks out following the path data/tracking',
        default=False)
    parser.add_argument(
        '-vo',
        '--video_out',
        help='Write tracking video out following the path data/tracking',
        default=False)
    parser.add_argument(
        '-db', '--dashboard', help='Send dashboard result', default=False)
    args = parser.parse_args()

    # Run
    if args.write_images == 'True':
        Config.Track.FACE_TRACK_IMAGES_OUT = True
    if args.video_out == 'True':
        Config.Track.TRACKING_VIDEO_OUT = True
    if args.dashboard == 'True':
        Config.SEND_QUEUE_TO_DASHBOARD = True
    Config.Track.PREDICT_DICT_OUT = True

    generate_video_sample(args.cam_url, args.area)
