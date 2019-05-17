import collections
import math
import torch
from torch.autograd import Variable
from distutils.version import LooseVersion
import numpy as np
import cv2
import os
from enum import IntEnum
from abc import ABCMeta, abstractmethod
import threading


def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scale the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r, c, *_ = im.shape
    ratio = targ / min(r, c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)


def crop(im, r, c, sz):
    '''
    crop image into a square of size sz, 
    '''
    return im[r:r + sz, c:c + sz]


def scale_to(x, ratio, targ):
    '''Calculate dimension of an image during scaling with aspect ratio'''
    return max(math.floor(x * ratio), targ)


class TfmType(IntEnum):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4


class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """

    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y = tfm_y
        self.store = threading.local()

    def set_state(self):
        pass

    def __call__(self, x, y):
        self.set_state()
        x, y = ((self.transform(x),
                 y) if self.tfm_y == TfmType.NO else self.transform(x, y) if
                self.tfm_y in (TfmType.PIXEL,
                               TfmType.CLASS) else self.transform_coord(x, y))
        return x, y

    def transform_coord(self, x, y):
        return self.transform(x), y

    def transform(self, x, y=None):
        x = self.do_transform(x, False)
        return (x, self.do_transform(y, True)) if y is not None else x

    @abstractmethod
    def do_transform(self, x, is_y):
        raise NotImplementedError


class CoordTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_square(y, x):
        r, c, *_ = x.shape
        y1 = np.zeros((r, c))
        y = y.astype(np.int)
        y1[y[0]:y[2], y[1]:y[3]] = 1.
        return y1

    def map_y(self, y0, x):
        y = CoordTransform.make_square(y0, x)
        y_tr = self.do_transform(y, True)
        return to_bb(y_tr)

    def transform_coord(self, x, ys):
        yp = partition(ys, 4)
        y2 = [self.map_y(y, x) for y in yp]
        x = self.do_transform(x, False)
        return x, np.concatenate(y2)


def center_crop(im, min_sz=None):
    """ Return a center crop of an image """
    r, c, *_ = im.shape
    if min_sz is None: min_sz = min(r, c)
    start_r = math.ceil((r - min_sz) / 2)
    start_c = math.ceil((c - min_sz) / 2)
    return crop(im, start_r, start_c, min_sz)


class CenterCrop(CoordTransform):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        sz: int
            size of the crop.
        tfm_y : TfmType
            type of y transformation.
    """

    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.min_sz, self.sz_y = sz, sz_y

    def do_transform(self, x, is_y):
        return center_crop(x, self.sz_y if is_y else self.min_sz)


class RandomCrop(CoordTransform):
    """ A class that represents a Random Crop transformation.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        targ: int
            target size of the crop.
        tfm_y: TfmType
            type of y transformation.
    """

    def __init__(self, targ_sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz, self.sz_y = targ_sz, sz_y

    def set_state(self):
        self.store.rand_r = random.uniform(0, 1)
        self.store.rand_c = random.uniform(0, 1)

    def do_transform(self, x, is_y):
        r, c, *_ = x.shape
        sz = self.sz_y if is_y else self.targ_sz
        start_r = np.floor(self.store.rand_r * (r - sz)).astype(int)
        start_c = np.floor(self.store.rand_c * (c - sz)).astype(int)
        return crop(x, start_r, start_c, sz)


class NoCrop(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ: int
            target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """

    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz, self.sz_y = sz, sz_y

    def do_transform(self, x, is_y):
        if is_y:
            return no_crop(
                x, self.sz_y, cv2.INTER_AREA
                if self.tfm_y == TfmType.PIXEL else cv2.INTER_NEAREST)
        else:
            return no_crop(x, self.sz, cv2.INTER_AREA)


class GoogleNetResize(CoordTransform):
    """ Randomly crops an image with an aspect ratio and returns a squared resized image of size targ 
    
    Arguments:
        targ_sz: int
            target size
        min_area_frac: float < 1.0
            minimum area of the original image for cropping
        min_aspect_ratio : float
            minimum aspect ratio
        max_aspect_ratio : float
            maximum aspect ratio
        flip_hw_p : float
            probability for flipping magnitudes of height and width
        tfm_y: TfmType
            type of y transform
    """

    def __init__(self,
                 targ_sz,
                 min_area_frac=0.08,
                 min_aspect_ratio=0.75,
                 max_aspect_ratio=1.333,
                 flip_hw_p=0.5,
                 tfm_y=TfmType.NO,
                 sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz, self.tfm_y, self.sz_y = targ_sz, tfm_y, sz_y
        self.min_area_frac, self.min_aspect_ratio, self.max_aspect_ratio, self.flip_hw_p = min_area_frac, min_aspect_ratio, max_aspect_ratio, flip_hw_p

    def set_state(self):
        # if self.random_state: random.seed(self.random_state)
        self.store.fp = random.random() < self.flip_hw_p

    def do_transform(self, x, is_y):
        sz = self.sz_y if is_y else self.targ_sz
        if is_y:
            interpolation = cv2.INTER_NEAREST if self.tfm_y in (
                TfmType.COORD, TfmType.CLASS) else cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_AREA
        return googlenet_resize(
            x,
            sz,
            self.min_area_frac,
            self.min_aspect_ratio,
            self.max_aspect_ratio,
            self.store.fp,
            interpolation=interpolation)


def compose(im, y, fns):
    """ Apply a collection of transformation functions :fns: to images """
    for fn in fns:
        #pdb.set_trace()
        im, y = fn(im, y)
    return im if y is None else (im, y)


class CropType(IntEnum):
    """ Type of image cropping. """
    RANDOM = 1
    CENTER = 2
    NO = 3
    GOOGLENET = 4


crop_fn_lu = {
    CropType.RANDOM: RandomCrop,
    CropType.CENTER: CenterCrop,
    CropType.NO: NoCrop,
    CropType.GOOGLENET: GoogleNetResize
}


class Transforms():

    def __init__(self,
                 sz,
                 tfms,
                 normalizer,
                 denorm,
                 crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO,
                 sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz, self.denorm, self.norm, self.sz_y = sz, denorm, normalizer, sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        self.tfms.append(crop_tfm)
        if normalizer is not None: self.tfms.append(normalizer)
        self.tfms.append(ChannelOrder(tfm_y))

    def __call__(self, im, y=None):
        return compose(im, y, self.tfms)

    def __repr__(self):
        return str(self.tfms)


class ChannelOrder():
    '''
    changes image array shape from (h, w, 3) to (3, h, w). 
    tfm_y decides the transformation done to the y element. 
    '''

    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y = tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        if self.tfm_y == TfmType.PIXEL: y = np.rollaxis(y, 2)
        elif self.tfm_y == TfmType.CLASS: y = y[..., 0]
        return x, y


def is_listy(x):
    return isinstance(x, (list, tuple))


def map_over(x, f):
    return [f(o) for o in x] if is_listy(x) else f(x)


IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
USE_GPU = torch.cuda.is_available()
print('Using CUDA device : ',torch.cuda.current_device())


def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x


def to_np(v):
    '''returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.'''
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list, tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v = v.data
    if torch.cuda.is_available():
        if isinstance(v, torch.cuda.HalfTensor): v = v.float()
    if isinstance(v, torch.FloatTensor): v = v.float()
    return v.cpu().numpy()


def V_(x, requires_grad=False, volatile=False):
    '''equivalent to create_variable, which creates a pytorch tensor'''
    return create_variable(x, volatile=volatile, requires_grad=requires_grad)


def V(x, requires_grad=False, volatile=False):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, lambda o: V_(o, requires_grad, volatile))


def create_variable(x, volatile, requires_grad=False):
    if type(x) != Variable:
        if IS_TORCH_04: x = Variable(T(x), requires_grad=requires_grad)
        else: x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x


def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor. 
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else:
            raise NotImplementedError(a.dtype)
    #if cuda: a = to_gpu(a, async=True)
    return a


def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]


def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            if str(fn).startswith("http"):
                req = urllib.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32) / 255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32) / 255
            if im is None: raise OSError('File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """

    def __init__(self, m, s):
        self.m = np.array(m, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)

    def __call__(self, x):
        return x * self.s + self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """

    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m = np.array(m, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)
        self.tfm_y = tfm_y

    def __call__(self, x, y=None):
        x = (x - self.m) / self.s
        if self.tfm_y == TfmType.PIXEL and y is not None:
            y = (y - self.m) / self.s
        return x, y


def image_gen(normalizer,
              denorm,
              sz,
              tfms=None,
              max_zoom=None,
              pad=0,
              crop_type=None,
              tfm_y=None,
              sz_y=None,
              pad_mode=cv2.BORDER_REFLECT,
              scale=None):
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y = TfmType.NO
    if tfms is None: tfms = []
    elif not isinstance(tfms, collections.Iterable): tfms = [tfms]
    if sz_y is None: sz_y = sz
    if scale is None:
        scale = [
            RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y)
            if max_zoom is not None else Scale(sz, tfm_y, sz_y=sz_y)
        ]
    elif not is_listy(scale):
        scale = [scale]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type != CropType.GOOGLENET: tfms = scale + tfms
    return Transforms(
        sz, tfms, normalizer, denorm, crop_type, tfm_y=tfm_y, sz_y=sz_y)


class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz: int
            target size to scale minimum size.
        tfm_y: TfmType
            type of y transformation.
    """

    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz, self.sz_y = sz, sz_y

    def do_transform(self, x, is_y):
        if is_y:
            return scale_min(
                x, self.sz_y, cv2.INTER_AREA
                if self.tfm_y == TfmType.PIXEL else cv2.INTER_NEAREST)
        else:
            return scale_min(x, self.sz, cv2.INTER_AREA)


def tfms_from_stats(stats,
                    sz,
                    aug_tfms=None,
                    max_zoom=None,
                    pad=0,
                    crop_type=CropType.RANDOM,
                    tfm_y=None,
                    sz_y=None,
                    pad_mode=cv2.BORDER_REFLECT,
                    norm_y=True,
                    scale=None):
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """

    #print(stats)

    if aug_tfms is None: aug_tfms = []
    tfm_norm = Normalize(
        *stats,
        tfm_y=tfm_y if norm_y else TfmType.NO) if stats is not None else None
    tfm_denorm = Denormalize(*stats) if stats is not None else None
    val_crop = CropType.CENTER if crop_type in (
        CropType.RANDOM, CropType.GOOGLENET) else crop_type
    val_tfm = image_gen(
        tfm_norm,
        tfm_denorm,
        sz,
        pad=pad,
        crop_type=val_crop,
        tfm_y=tfm_y,
        sz_y=sz_y,
        scale=scale)
    trn_tfm = image_gen(
        tfm_norm,
        tfm_denorm,
        sz,
        pad=pad,
        crop_type=crop_type,
        tfm_y=tfm_y,
        sz_y=sz_y,
        tfms=aug_tfms,
        max_zoom=max_zoom,
        pad_mode=pad_mode,
        scale=scale)
    return trn_tfm, val_tfm
