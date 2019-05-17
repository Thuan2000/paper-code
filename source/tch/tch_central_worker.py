import argparse
import time
from matcher import KdTreeMatcher
from config import Config
from rabbitmq import RabbitMQ
from tracker import TrackersList

# TODO (@man): move this to folder tch


def central_function():
    rabbit_mq = RabbitMQ((Config.Rabbit.USERNAME, Config.Rabbit.PASSWORD),
                         (Config.Rabbit.IP_ADDRESS, Config.Rabbit.PORT))
    cen_list = TrackersList()
    matcher = KdTreeMatcher()
    matcher._match_case = 'TCH'

    while True:
        fid, image, emb, area, _, origin_bb, angle = \
                    rabbit_mq.receive_tracking(rabbit_mq.RECEIVE_CAM_WORKER_TRACKING_QUEUE)
        if fid is not None:
            print("-- Received a face-track message from cam server")
            generated_fid = str(fid) + '-' + area
            cen_list.update_trackers_list(generated_fid, origin_bb, image, emb,
                                          angle, area, 0, matcher, rabbit_mq)

        # check time
        cen_list.check_delete_trackers(matcher, rabbit_mq)

        # Check extract trackers history time
        # cen_list.trackers_history.check_time()

        if fid is None:
            print('Waiting for new job ...')
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'TCH project', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-wi',
        '--write_images',
        help='Write all face-tracks out following the path data/tracking',
        default=False)
    args = parser.parse_args()
    if args.write_images == 'True':
        Config.Track.FACE_TRACK_IMAGES_OUT = True
    central_function()
