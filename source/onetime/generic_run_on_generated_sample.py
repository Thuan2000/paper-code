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
import time
from frame_reader import URLFrameReader, RabbitFrameReader
from matcher import KdTreeMatcher
from config import Config
from rabbitmq import RabbitMQ
from video_writer import VideoHandle
from tracker import TrackersList, TrackerResultsDict
from cv_utils import FaceAngleUtils, CropperUtils, PickleUtils, clear_tracking_folder, is_inner_bb


class FrameSample:

    def __init__(self):
        self.frame_id = 0
        self.origin_bbs = []
        self.points = []
        self.read_image = []
        self.embs = []


rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                     (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))


def generic_function(cam_url, area):
    '''
    This is main function
    '''
    print("Generic function")
    print("Cam URL: {}".format(cam_url))
    print("Area: {}".format(area))
    # Variables for tracking faces

    # Variables holding the correlation trackers and the name per faceid
    list_of_trackers = TrackersList()

    clear_tracking_folder()

    matcher = KdTreeMatcher()
    print("Load sample")
    frame_sample = PickleUtils.read_pickle('../session/db/sample.pkl')
    frame_counter = 0
    track_results = TrackerResultsDict()
    predict_dict = {}
    if Config.CALC_FPS:
        start_time = time.time()
    if args.cam_url is not None:
        frame_reader = URLFrameReader(args.cam_url, scale_factor=1)
    else:
        frame_reader = RabbitFrameReader(rabbit_mq)
    video_out = None
    video_out_fps = 24
    video_out_w = 1920
    video_out_h = 1080
    print(video_out_fps, video_out_w, video_out_h)
    center = (int(video_out_w / 2), int(video_out_h / 2))
    bbox = [
        int(center[0] - 0.35 * video_out_w),
        int(center[1] - video_out_h * 0.35),
        int(center[0] + 0.35 * video_out_w),
        int(center[1] + 0.35 * video_out_h)
    ]
    if Config.Track.TRACKING_VIDEO_OUT:
        video_out = VideoHandle('../data/tracking_video_out.avi', video_out_fps,
                                int(video_out_w), int(video_out_h))
    try:
        while True:  # frame_reader.has_next():
            frame = frame_sample[frame_counter].read_image
            if frame is None:
                print("Waiting for the new image")
                trackers_return_dict, predict_trackers_dict = \
                    list_of_trackers.check_delete_trackers(matcher, rabbit_mq)
                track_results.update_two_dict(trackers_return_dict)
                predict_dict.update(predict_trackers_dict)
                continue

            print("Frame ID: %d" % frame_counter)
            print('Num of ids in matcher: {}'.format(matcher._numofids))

            if Config.Track.TRACKING_VIDEO_OUT:
                video_out.tmp_video_out(frame)
            if Config.CALC_FPS:
                fps_counter = time.time()

            list_of_trackers.update_dlib_trackers(frame)

            if frame_counter % Config.Frame.FRAME_INTERVAL == 0:
                origin_bbs = frame_sample[frame_counter].origin_bbs
                points = frame_sample[frame_counter].points
                for i, origin_bb in enumerate(origin_bbs):
                    print(is_inner_bb(bbox, origin_bb))
                    if not is_inner_bb(bbox, origin_bb):
                        continue
                    display_face, _ = CropperUtils.crop_display_face(
                        frame, origin_bb)

                    # Calculate embedding
                    emb_array = frame_sample[frame_counter].embs[i]

                    # Calculate angle
                    angle = FaceAngleUtils.calc_angle(points[:, i])

                    # TODO: refractor matching_detected_face_with_trackers
                    matched_fid = list_of_trackers.matching_face_with_trackers(
                        frame, origin_bb, emb_array)

                    # Update list_of_trackers
                    list_of_trackers.update_trackers_list(
                        matched_fid, time.time(), origin_bb, display_face,
                        emb_array, angle, area, frame_counter, i, matcher,
                        rabbit_mq)

            trackers_return_dict, predict_trackers_dict = \
                list_of_trackers.check_delete_trackers(matcher, rabbit_mq)
            track_results.update_two_dict(trackers_return_dict)
            predict_dict.update(predict_trackers_dict)

            # Check extract trackers history time (str(frame_counter) + '_' + str(i))
            list_of_trackers.trackers_history.check_time()

            frame_counter += 1
            if Config.CALC_FPS:
                print("FPS: %f" % (1 / (time.time() - fps_counter)))
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
        if Config.Track.PREDICT_DICT_OUT:
            PickleUtils.save_pickle(Config.PREDICTION_DICT_FILE, predict_dict)
    except KeyboardInterrupt:
        print('Keyboard Interrupt !!! Release All !!!')
        if Config.CALC_FPS:
            print('Time elapsed: {}'.format(time.time() - start_time))
            print('Avg FPS: {}'.format(
                (frame_counter + 1) / (time.time() - start_time)))
        frame_reader.release()
        if Config.Track.TRACKING_VIDEO_OUT:
            print('Write track video')
            video_out.write_track_video(track_results.tracker_results_dict)
            video_out.release()
        if Config.Track.PREDICT_DICT_OUT:
            PickleUtils.save_pickle(Config.PREDICTION_DICT_FILE, predict_dict)


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
    parser.add_argument(
        '-mo',
        '--min_dist_out',
        help='write tracking folder with good element n min distance',
        default=False)
    args = parser.parse_args()

    # Run
    if args.write_images == 'True':
        Config.Track.FACE_TRACK_IMAGES_OUT = True
    if args.video_out == 'True':
        Config.Track.TRACKING_VIDEO_OUT = True
    if args.dashboard == 'True':
        Config.SEND_QUEUE_TO_DASHBOARD = True
    if args.min_dist_out == 'True':
        Config.Track.MIN_MATCH_DISTACE_OUT = True
    Config.Track.PREDICT_DICT_OUT = True

    generic_function(args.cam_url, args.area)
