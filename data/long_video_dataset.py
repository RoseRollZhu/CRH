import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class DataPair(object):
    def __init__(self, root_image, input_frame, root_label=None, gen_frame=None, islabeldir_absolute=False):
        self.input_frames = [os.path.join(root_image, i) for i in input_frame]
        if gen_frame is not None:
            self.gen_frames = [os.path.join(root_image, i) for i in gen_frame]
        else:
            self.gen_frames = None
        if root_label is not None:
            if not islabeldir_absolute:
                self.gen_frames_label = [os.path.join(
                    root_label, i) for i in gen_frame]
            else:
                self.gen_frames_label = root_label
        else:
            self.gen_frames_label = None


def toint64(x):
    return torch.tensor(np.asarray(x).astype(np.int64))

class LongVideoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.generate_length = opt.n_frame * opt.n_recurrent
        self.data = []
        if 'camvid' in opt.dataroot:
            self.condition_length = 4
            self.transform_image = transforms.Compose([
                transforms.Resize((256, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.transform_label = transforms.Compose([
                transforms.Resize((256, 512)),
                # transforms.Lambda(lambda x: torch.tensor(
                #     np.asarray(x).astype(np.int64).copy()))
            ])
            for term in ['0001', '0005', '0006', '0016']:
                images = os.listdir(os.path.join(opt.dataroot, 'image', term))
                images.sort(key=lambda x: int(x.split('_')[1][1:]))
                if opt.phase == 'train':
                    images = images[:(len(images)//6*5)]
                else:
                    images = images[(len(images)//6*5):]

                for i in range(len(images)):
                    if i + self.condition_length + self.generate_length - 1 >= len(images):
                        break
                    datapair = DataPair(
                        os.path.join(opt.dataroot, 'image', term),
                        images[i:i+self.condition_length],
                        os.path.join(opt.dataroot, 'label_id',
                                     term) if opt.isTrain else None,
                        images[i+self.condition_length:i+self.condition_length +
                               self.generate_length] if opt.isTrain else None,
                    )
                    self.data.append(datapair)

        elif 'kitti' in opt.dataroot:
            self.condition_length = 4
            self.transform_image = transforms.Compose([
                transforms.Resize((256, 832)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.transform_label = transforms.Compose([
                transforms.Resize((256, 832)),
                # transforms.Lambda(lambda x: torch.from_numpy(
                #     np.asarray(x).astype(np.int64).copy()))
            ])
            if opt.phase == 'train':
                sequences = os.listdir(opt.dataroot)
                sequences.remove('2011_09_26_drive_0060_sync')
                sequences.remove('2011_09_26_drive_0084_sync')
                sequences.remove('2011_09_26_drive_0093_sync')
                sequences.remove('2011_09_26_drive_0096_sync')
            else:
                sequences = [
                    '2011_09_26_drive_0060_sync',
                    '2011_09_26_drive_0084_sync',
                    '2011_09_26_drive_0093_sync',
                    '2011_09_26_drive_0096_sync'
                ]
            sequences.sort(key=lambda x: (
                int(x.split('_')[1]), int(x.split('_')[4])))
            for seq in sequences:
                images = os.listdir(os.path.join(
                    opt.dataroot, seq, 'image_02', 'data'))
                images.sort()
                for i in range(len(images)):
                    if i + self.condition_length + self.generate_length - 1 >= len(images):
                        break
                    datapair = DataPair(
                        os.path.join(opt.dataroot, seq, 'image_02', 'data'),
                        images[i:i+self.condition_length],
                        os.path.join(opt.dataroot, seq, 'label_id'),
                        images[i+self.condition_length:i +
                               self.condition_length+self.generate_length],
                    )
                    self.data.append(datapair)

        elif 'cityscapes' in opt.dataroot:
            self.condition_length = 4
            self.transform_image = transforms.Compose([
                transforms.Resize((256, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.transform_label = transforms.Compose([
                transforms.Resize((256, 512)),
                # transforms.Lambda(lambda x: torch.from_numpy(
                #     np.asarray(x).astype(np.int64).copy()))
            ])
            sequences = os.listdir(os.path.join(
                opt.dataroot, 'leftImg8bit', opt.phase))
            for seq in sequences:
                images = os.listdir(os.path.join(
                    opt.dataroot, 'leftImg8bit', opt.phase, seq))
                images.sort(key=lambda x: (
                    int(x.split('_')[1]), int(x.split('_')[2])))
                for image in images:
                    split_str = image.split('_')
                    front_str = split_str[0] + '_' + split_str[1]
                    back_str = split_str[3]
                    index = int(split_str[2])
                    start_index = index - 19
                    end_index = index + 10
                    related_images = [
                        front_str + '_%06d_' % (idx) + back_str for idx in range(start_index, end_index+1)]
                    related_labels = [os.path.join(opt.dataroot, 'gtFine_sequence', opt.phase, seq, rimg.replace(
                        'leftImg8bit', 'labelIds')) for rimg in related_images]
                    if opt.phase == 'train':
                        for i in range(len(related_images)):
                            if i + self.condition_length + self.generate_length - 1 >= len(related_images):
                                break
                            datapair = DataPair(
                                os.path.join(
                                    opt.dataroot, 'leftImg8bit_sequence', opt.phase, seq),
                                related_images[i:i+self.condition_length],
                                related_labels[i+self.condition_length:i +
                                               self.condition_length+self.generate_length],
                                related_images[i+self.condition_length:i +
                                               self.condition_length+self.generate_length],
                                True
                            )
                            self.data.append(datapair)
                    else:
                        datapair = DataPair(
                            os.path.join(
                                opt.dataroot, 'leftImg8bit_sequence', opt.phase, seq),
                            related_images[:self.condition_length],
                            related_labels[self.condition_length:self.condition_length +
                                           self.generate_length],
                            related_images[self.condition_length:self.condition_length +
                                           self.generate_length],
                            True
                        )
                        self.data.append(datapair)

        


        if len(self.data) > opt.max_dataset_size:
            self.data = self.data[:opt.max_dataset_size]

        self.dataset_size = len(self.data)

        self.batchSize = opt.batchSize

    def __getitem__(self, index):
        dataPair = self.data[index]
        flip = random.random() > 0.5 if self.opt.isTrain and not self.opt.no_flip else False

        input_images = []
        for i in dataPair.input_frames:
            img = Image.open(i).convert('RGB')
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.transform_image(img)
            input_images.append(img.unsqueeze(1))
        input_images = torch.cat(input_images, 1)

        if dataPair.gen_frames is not None:
            gen_images = []
            for i in dataPair.gen_frames:
                img = Image.open(i).convert('RGB')
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = self.transform_image(img)
                gen_images.append(img.unsqueeze(1))
            gen_images = torch.cat(gen_images, 1)
        else:
            gen_images = -1

        if dataPair.gen_frames_label is not None:
            gen_segs = []
            for i in dataPair.gen_frames_label:
                img = Image.open(i)
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = self.transform_label(img)
                img = toint64(img)
                gen_segs.append(img.unsqueeze(0))
            gen_segs = torch.cat(gen_segs, 0)
        else:
            gen_segs = -1

        input_dict = {
            'input_images': input_images,
            'gen_images': gen_images,
            'gen_segs': gen_segs,
            'path': dataPair.input_frames[0],
        }
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'LongVideoDataset'
