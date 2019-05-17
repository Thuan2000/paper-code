import numpy as np
import itertools
import argparse
import sys
import os
import cv2
import prod.detect_occlusion as do
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

URL = 'https://www.dropbox.com/s/ognh8tgzfb9f0dx/occlusion_dataset.tar.gz?dl=0'


def fetch_data(data_dir='../../data/occlusion/Image'):
    if not os.listdir(data_dir):
        print('Downloading ...')
        data_path = os.path.join(data_dir, 'dataset.tar.gz')
        data_url = URL
        download = "wget '{0}' -O {1}".format(data_url, data_path)
        extract = "tar -xvf {0} -C {1}".format(data_path, data_dir)
        remove = "rm {}".format(data_path)
        os.system(download)
        os.system(extract)
        os.system(remove)


def get_all_files(root):
    for path, subdirs, files in os.walk(root):
        for name in files:
            yield path, name


def get_label(name):
    if name == 'noocclusion':
        return do.NO_OCCLUSION
    elif name == 'mask':
        return do.NO_EYES
    elif name == 'glasses':
        return do.NO_MOUTH


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def extract_faces(data_dir, output_dir):
    face_detector = do.MTCNNFaceDetection()
    if os.path.exists(data_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        count = 0
        for file in os.listdir(data_dir):
            image = cv2.imread(os.path.join(data_dir, file))
            faces = face_detector.detect(image)
            for face in faces:
                _, face_color = do.get_face_bound(face, image, image)
                cv2.imwrite(
                    os.path.join(output_dir, "{}.jpg".format(count)),
                    face_color)
                if count % 10 == 0:
                    print(count)
                count += 1


def evaluate_detector(data_dir, detector, output_dir):
    if os.path.exists(data_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(os.path.join(output_dir, "True")):
            os.mkdir(os.path.join(output_dir, "True"))
        if not os.path.exists(os.path.join(output_dir, "False")):
            os.mkdir(os.path.join(output_dir, "False"))

        corect = 0
        total = 0
        count = 0
        for file in os.listdir(data_dir):
            image = cv2.imread(os.path.join(data_dir, file))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eyes = detector.detect(gray)
            for eye in eyes:
                clone_image = image.copy()
                do.draw_box(clone_image, eye, do.GREEN, 1)
                while (1):
                    cv2.imshow("Result", clone_image)
                    k = cv2.waitKey(0)
                    if k == ord('z'):
                        # correct detected
                        corect += 1
                        file_name = os.path.join(output_dir, "True",
                                                 "{}.jpg".format(count))
                        cv2.imwrite(file_name, clone_image)
                        count += 1
                        break
                    elif k == ord('x'):
                        file_name = os.path.join(output_dir, "False",
                                                 "{}.jpg".format(count))
                        cv2.imwrite(file_name, clone_image)
                        count += 1
                        break
                    elif k == 27:
                        sys.exit()
                total += 1
        print("Precision {}".format(float(corect / total)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--d', dest='data_dir', default='../data/occlusion/Image')
    parser.add_argument('--m', dest='model_dir', default='../models/opencv/')
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    # fetch_data(data_dir)

    test = []
    pred = []
    checker = do.OcclusionDetection(model_dir)
    for i, (path, name) in enumerate(get_all_files(data_dir)):
        if i % 10 == 0:
            print('Images %d' % i)
        label = get_label(name.split('_')[0])
        image = cv2.imread(os.path.join(path, name))
        result = checker.detect(image)
        test.append(label)
        pred.append(result)

    class_names = ['no occlusion', 'no mouth', 'no eyes', 'no face']
    cnf_matrix = confusion_matrix(test, pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        title='Confusion matrix, without normalization')

    plt.show()
