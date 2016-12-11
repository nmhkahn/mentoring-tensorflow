import skimage.io
import numpy as np
from skimage.transform import resize

def load_and_preprocess_image( path, 
                               resize_dim=None ):
    
    MEAN_VALUE = np.array([123.68, 116.779, 103.939])

    im = load_image(path)
    if not resize_dim is None:
        im = resize_image(im, resize_dim)
    
    im *= 255
    im -= MEAN_VALUE

    return im


# actually this function is based on caffe.io :)
def load_image(filename, color=True):
    
    img = skimage.img_as_float(skimage.io.imread(filename, 
        as_grey=not color)).astype(np.float32)
    # in case image is grey-scale
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        # if color
        # then convert grey-scale into RGB scale
        if color:
            img = np.tile(img, (1, 1, 3))
    # in case image channels are RGBA, ignore A(transparency)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):

    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)