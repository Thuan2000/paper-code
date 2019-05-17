from cv_utils import get_avg_dists, calc_iou
import random
import unittest

class TestCVUtils(unittest.TestCase):
    def test_get_avg_dists(self):
        labels = []
        dists = []
        result = {}
        nof_labels = random.randint(10000, 100000)
        for i in range(nof_labels):
            label = random.randint(0, nof_labels)
            labels.append(label)
            dist = random.random()
            dists.append(dist)
            if not label in result:
                result[label] = []
            result[label].append(dists[i])
        for fid in result:
            nof_this_fid = len(result[fid])
            result[fid] = {
                'dist': sum(result[fid]) / len(result[fid]),
                'rate': nof_this_fid / nof_labels
            }
        result2 = get_avg_dists(labels, dists)
        self.assertEqual(result == result2, 1)

    def bb_intersection_over_union_by_Adrian_Rosebrock(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def test_calc_iou(self):
        bb1 = [random.randint(0, 50), random.randint(0, 50), random.randint(50, 2000), random.randint(50, 2000)]
        bb2 = [random.randint(0, 50), random.randint(0, 50), random.randint(50, 2000), random.randint(50, 2000)]
        self.assertEqual(int((self.bb_intersection_over_union_by_Adrian_Rosebrock(bb1, bb2) - calc_iou(bb1, bb2)) * 100), 0)

if __name__ == '__main__':
    unittest.main()
