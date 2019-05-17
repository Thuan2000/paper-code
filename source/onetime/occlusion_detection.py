import cv2
import sys
import numpy as np
import argparse
import prod.detect_occlusion as do
from frame_reader import URLFrameReader


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
            frame_label = do.process_result(detected_result)
            if frame_label == do.NO_OCCLUSION:
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
            do.show_information(frame, curent_frame, frames_per_state,
                                state_validation)
            detected_result = detector.detect(frame)
            frame_label = do.process_result(detected_result)
            if frame_label == do.NO_OCCLUSION:
                state_correct += 1

            if curent_frame >= frames_per_state:
                state_validation = True if state_correct >= 1 else False
                curent_frame = 0
                state_correct = 0
            do.display_result_board(result_board, detected_result)
            cv2.imshow('frame', frame)
            cv2.imshow('result', result_board)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', dest='cam_id', default=0)
    parser.add_argument('--m', dest='model_dir', default=do.OPENCV_DIR)
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    # Arguments
    args = parse_arguments(sys.argv[1:])
    cam_id = args.cam_id
    model_dir = args.model_dir
    occlusion_detection = do.OcclusionDetection(model_dir)
    occlusion_dection_video(cam_id, occlusion_detection)
