'''
Video Writer
'''
import os
import cv2
from config import Config


class VideoHandle:
    '''
    This class for writing output video
    '''

    def __init__(self, write_time, fps, w_video, h_video):
        # Define the codec and create VideoWriter object

        self.out_path = os.path.join(
            Config.Track.VIDEO_OUT_PATH,
            '{}_tracking_video_out.avi'.format(write_time))
        self.out_path = 'test.avi'
        self.tmp_path = os.path.join(
            Config.Track.VIDEO_OUT_PATH,
            '{}_tmp_video_for_tracking.avi'.format(write_time))
        self.tmp_path = 'tmp_test.avi'
        self.clear_video()
        self.out = cv2.VideoWriter(self.out_path,
                                   cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (w_video, h_video))
        self.tmp_out = cv2.VideoWriter(self.tmp_path,
                                       cv2.VideoWriter_fourcc(*'XVID'), fps,
                                       (w_video, h_video))

    def write_track_video(self, track_results_dict, database):
        '''
        Write recognized tracking video
        '''
        print('Writing video ...')
        self.tmp_out.release()
        frame_reader = cv2.VideoCapture(self.tmp_path)
        frame_counter = 0
        while True:
            _, frame = frame_reader.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame is None:
                break
            print('Write Frame: %d' % frame_counter)
            if frame_counter in track_results_dict.keys():
                for i, name in enumerate(
                        track_results_dict[frame_counter].track_names):

                    visitor = database.mongodb_db['visitor'].find({
                        'visitorId':
                        name
                    })
                    if visitor.count() > 0:
                        displayName = visitor[0]['displayName']
                        if len(displayName) > 0:
                            name = displayName

                    print(name)
                    bb0 = int(
                        track_results_dict[frame_counter].bounding_boxes[i][0])
                    bb1 = int(
                        track_results_dict[frame_counter].bounding_boxes[i][1])
                    bb2 = int(
                        track_results_dict[frame_counter].bounding_boxes[i][2])
                    bb3 = int(
                        track_results_dict[frame_counter].bounding_boxes[i][3])
                    cv2.rectangle(frame, (bb0, bb1), (bb2, bb3), (0, 165, 255),
                                  2)

                    cv2.putText(
                        frame, str(name), (int(bb0 + (bb2 - bb0) / 2), bb1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            self.out.write(frame)
            frame_counter += 1
        print('Video has been written as ' + self.out_path)
        # os.remove(self.tmp_path)

    def tmp_video_out(self, frame):
        '''
        Write temporary video
        '''
        self.tmp_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release_tmp(self):
        self.tmp_out.release()

    def release_out(self):
        self.out.release()

    def clear_video(self):
        '''
        Remove tracking video and temporary video
        '''
        if os.path.isfile(self.out_path):
            os.remove(self.out_path)
        if os.path.isfile(self.tmp_path):
            os.remove(self.tmp_path)
