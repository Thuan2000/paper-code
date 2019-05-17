from config import Config
import cv2


class ROIFrameProcessor:

    def __init__(self, scale_factor=1, roi_ratio=Config.Frame.ROI_CROP):
        # assert scale_factor > 1
        self._scale_factor = scale_factor
        self._roi_ratio = roi_ratio
        self._roi = None

    def preprocess(self, frame):
        # Rescale frame and crop ROI for faster face detection

        # Crop ROI
        if self._roi is None:
            self._roi = ROIFrameProcessor._crop_roi(frame, self._roi_ratio)
        roi = self._roi
        frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

        # Rescale frame
        frame = cv2.resize(frame, (int(len(frame[0]) / self._scale_factor),
                                   int(len(frame) / self._scale_factor)))
        return frame

    @staticmethod
    def _crop_roi(frame, roi_ratio):
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        roi = [
            int(center[0] - roi_ratio[0] * w),
            int(center[1] - roi_ratio[1] * h),
            int(center[0] + roi_ratio[2] * w),
            int(center[1] + roi_ratio[3] * h)
        ]
        return roi

    def postprocess(self, bbs, pts):
        # map bb and pt to original frame
        if len(bbs) == 0:
            # no face detected, return
            return bbs, pts

        # Rescale frame
        bbs, pts = bbs * self._scale_factor, pts * self._scale_factor

        # Crop ROI
        roi = self._roi
        bbs[:, 0] += roi[0]
        bbs[:, 2] += roi[0]
        bbs[:, 1] += roi[1]
        bbs[:, 3] += roi[1]
        print('Landmark', pts.shape)
        pts[0:5, :] += roi[0]
        pts[5:10, :] += roi[1]
        return bbs, pts
