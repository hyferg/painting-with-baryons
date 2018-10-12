import numpy as np

def SetTransform(tensor, save_dirs, save=False):
        _tensors = [np.zeros(shape=tensor.shape, dtype=tensor.dtype) for _ in save_dirs]
        for tens in _tensors:
                print(tens.shape)
        for i in range(0, len(tensor)):
                if i % 10 == 0:
                        print(i)
                _mod = FourierSplit(tensor[i])
                for j, split in enumerate(FourierSplit(tensor[i])):
                        _tensors[j][i] = split

        if save:
                for i, save_dir in enumerate(save_dirs):
                        np.save(save_dir, _tensors[i])
        return _tensors

def FourierSplit(tensor, half_box=30):
        mask = circ_mask(tensor.shape[-2:], half_box)
        mask_inv = 1 - mask

        box_small = apply_mask(mask)
        box_large = apply_mask(mask_inv)

        _apply = [
            np.fft.fft2,
            np.fft.fftshift,
        ]

        small_apply = _apply + [box_small]
        large_apply = _apply + [box_large]

        revert = [
            np.fft.ifftshift,
            np.fft.ifft2,
            np.float32,
        ]

        return apply_split(tensor, [small_apply, large_apply], revert)


def apply_split(tensor, splits, revert):
    masks = [flist(tensor, split) for split in splits]
    reverts = [flist(mask, revert) for mask in masks]
    return reverts


def flist(x, transforms):
    for func in transforms:
        x = func(x)
    return x

def log_norm(img):
    img = np.float32(np.abs(img))
    c = 255.0/np.log(1+np.max(np.abs(img)))
    for y in np.nditer(img, op_flags=['readwrite']):
        val = c*np.log(1+np.abs(y))
        y *= 0
        y += val
    return img

def apply_mask(mask):
    def inner(img):
        return img*mask
    return inner

def circ_mask(shape, radius):
    a = np.zeros(shape)
    cy, cx = a.shape[0]//2, a.shape[1]//2
    x, y = np.ogrid[-radius:radius,-radius:radius]
    index = x**2 + y**2 <= radius**2
    a[cy-radius:cy+radius,cx-radius:cx+radius][index] = 1
    return a
