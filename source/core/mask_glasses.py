from mask_glasses_classifier.normalize_and_predict import *
from config import Config


class GlassesClassifier():
    def __init__(self, path_model=Config.Model.GLASSES_DIR):
        self.model_predict = torch.load(path_model, map_location='cuda:0') # Change map_location='cpu' if use cpu

        model_resnet34 = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        trn_tfms, self.val_tfms = tfms_from_stats(model_resnet34, 224)

    def is_wearing_glasses(self, images):
        """
        Predict array images return Bool array

        Arguments:
            images: normalize images array
        """

        model_predict = self.model_predict
        normImg = self.val_tfms

        if len(images.shape) == 3:
            arrImg = normImg(images)[None]
        else:
            arrImg = np.asarray([normImg(img) for img in images])

        model_predict.eval()

        result = to_np(model_predict(to_gpu(V(T(arrImg))))) # Remove to_gpu function if use CPU

        preds = [bool(1 - np.argmax(pred)) for pred in result]

        return preds


class MaskClassifier():
    def __init__(self, path_model=Config.Model.MASK_DIR):
        self.model_predict = torch.load(path_model, map_location='cuda:0')

        model_resnet34 = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        trn_tfms, self.val_tfms = tfms_from_stats(model_resnet34, 224)

    def is_wearing_mask(self, images):
        """
        Predict array images return Bool array

        Arguments:
            images: normalize images array
        """

        model_predict = self.model_predict
        normImg = self.val_tfms

        if len(images.shape) == 3:
            arrImg = normImg(images)[None]
        else:
            arrImg = np.asarray([normImg(img) for img in images])

        model_predict.eval()

        result = to_np(model_predict(to_gpu(V(T(arrImg))))) # Remove to_gpu function if use CPU

        preds = [bool(1 - np.argmax(pred)) for pred in result]

        return preds
