from mask_glasses_classifier.normalize_and_predict import *
from config import Config

# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class MasanCustomerClassification():
    def __init__(self, path_model = Config.Model.MASAN_CUSTOMER_DIR, use_cuda = False):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model_predict = torch.load(path_model, map_location='cuda:0') # Change map_location='cpu' if use cpu
        else:
            self.model_predict = torch.load(path_model, map_location='cpu')

        model_resnet34 = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        trn_tfms, self.val_tfms = tfms_from_stats(model_resnet34, 224)

    def preprocessing(self, image):
        img_norm = image.astype(np.float32)/255
        img_preprocessed = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
        return img_preprocessed

    def predict(self, images):
        """
        Predict array images return Bool array

        Arguments:
            images: normalize images array
        """
        model_predict = self.model_predict
        normImg = self.val_tfms

        images = self.preprocessing(images)

        if len(images.shape) == 3:
            arrImg = normImg(images)[None]
        else:
            arrImg = np.asarray([normImg(img) for img in images])

        with torch.no_grad():
            model_predict.eval()
            if self.use_cuda:
                result = to_np(model_predict(to_gpu(V(T(arrImg))))) # Remove to_gpu function if use CPU
            else:
                result = to_np(model_predict((V(T(arrImg)))))

        preds = [bool(1 - np.argmax(pred)) for pred in result]
        predicts = np.exp(result)[0]
        predict = predicts.argmax()

        # class labels: {
        # 0: massan_staff
        # 1: security_staff
        # 2: customer
        # }
        if predict == 2:
            return True
        return False

