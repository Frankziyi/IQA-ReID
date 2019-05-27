from __future__ import print_function, division
from __future__ import absolute_import

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions, CAM=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target == '-1' or target == '0':
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    if CAM :
                        cam = fname.split('c', 1)[1][0]
                        item = (path, class_to_idx[target], int(target), int(cam))
                    else:
                        item = (path, class_to_idx[target], int(target))
                    images.append(item)

    return images

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class Dataset(data.Dataset):
    def __init__(self, image_dir, transform=None, CAM=False):
        self.transform = transform
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        if self.CAM:
            path, pid, real_id, cam = self.data[index]
        else:
            path, pid, real_id = self.data[index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.CAM:
            return img, pid, real_id, cam
        else:
            return img, pid, real_id

    def __len__(self):
        return len(self.data)
