from __future__ import print_function
import re
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import numpy as np
import os
import argparse
import dill as pickle
import cv2


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
@torch.no_grad()
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, size=None):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype) if size is None else cv2.resize(image_numpy.astype(imtype), size)


# Converts a one-hot tensor into a colorful label map
@torch.no_grad()
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    elif n_label == 2:
        label_tensor = label_tensor.cpu().float()
        label_tensor = label_tensor.max(0, keepdim=True)[1]
        label_tensor = label_tensor * 255
        label_numpy = np.transpose(label_tensor.repeat(3, 1, 1).numpy(), (1, 2, 0))
        return label_numpy.astype(imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


@torch.no_grad()
def tensor2seq(image_tensor, normalize=True, size=None):
    b = image_tensor.size(0)
    image_tensor = image_tensor.permute(0, 2, 3, 1).float()
    if normalize:
        image_tensor = (image_tensor + 1.0) / 2.0 * 255.0
    else:
        image_tensor = image_tensor * 255.0
    image_numpy = image_tensor.clamp(min=0, max=255).to(torch.uint8).cpu().numpy()
    img_set = []
    for i in range(b):
        img_set.append(cv2.resize(image_numpy[i], size) if size is not None else image_numpy[i])
    return img_set


def tensorColor(label_tensor, n_label):
    return Colorize(n_label).ColorTensor(label_tensor)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_gif(image_set, gif_path, fps=5):
    image_set = [Image.fromarray(x).convert('RGB') for x in image_set]
    image_set[0].save(gif_path, save_all=True, append_images=image_set[1:], loop=0, duration=1./fps)


def save_mp4(image_set, vid_path, fps=5, size=(128,128)):
    videowriter = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    for img in image_set:
        videowriter.write(img[:, :, ::-1])


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.state_dict(), save_path)


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    print(f'Loading {save_filename}....')
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path, map_location=torch.device('cpu'))
    net.load_state_dict(weights, strict=False)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

    def ColorTensor(self, image):
        b, h, w = image.size()
        color_image_r = torch.zeros(b, h, w, device=image.device)
        color_image_g = torch.zeros(b, h, w, device=image.device)
        color_image_b = torch.zeros(b, h, w, device=image.device)
        for label in range(0, len(self.cmap)):
            mask = image == label
            r, g, b = self.cmap[label].float()
            color_image_r[mask] = r.item()
            color_image_g[mask] = g.item()
            color_image_b[mask] = b.item()
        color_image = torch.cat((color_image_r.unsqueeze(1), color_image_g.unsqueeze(1), color_image_b.unsqueeze(1)), 1)
        color_image = color_image / (255.0 / 2.0) - 1.0
        return color_image
