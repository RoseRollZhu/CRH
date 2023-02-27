import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from transformers import T5Tokenizer, T5Model


def toint64(x):
    return torch.tensor(np.asarray(x).astype(np.int64))

class text_to_video_dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.generate_length = opt.n_frame * opt.n_recurrent
        self.n_recurrent = opt.n_recurrent
        self.data = []
        self.transform_image = transforms.Compose([
                transforms.Resize((opt.display_height, opt.display_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # input frame
        self.condition_length = opt.condition_length
        # output frame
        self.n_frame = opt.n_frame
    
        error_num = 0
        self.videofilelist = []
        get_video_num = 0

        if opt.read_from_npy == 1:
            # use pretrained npy file to boost the data process
            temp = np.load(os.path.join(opt.dataroot, 'data.npy'), allow_pickle=True)
            with tqdm(total=temp.shape[0]) as progress_bar:
                for i in range(temp.shape[0]):
                    self.data.append(temp[i])
                    progress_bar.update(1)

        else:
            if 'Kinetics' in opt.dataroot:
                if opt.phase == 'train':
                    file = open(os.path.join(opt.dataroot, 'train.csv'))
                    total_count = len(file.readlines())
                    file = open(os.path.join(opt.dataroot, 'train.csv'))
                else:
                    file = open(os.path.join(opt.dataroot, 'test.csv'))
                    total_count = len(file.readlines())
                    file = open(os.path.join(opt.dataroot, 'test.csv'))
                
                with tqdm(total=total_count) as progress_bar:
                    for each_file_line in file:
                        text_condition, id, start_frame, end_frame, divide = each_file_line.split(',')
                        divide = divide.split('\n')[0]
                        if '"' in text_condition:
                            _, text_condition, _ = text_condition.split('"')
                        this_video_path = os.path.join(opt.dataroot, 
                                                        divide,
                                                        f'{id}_{str(start_frame).zfill(6)}_{str(end_frame).zfill(6)}.mp4')
                        if not os.path.exists(this_video_path):
                            print(f'[warning] Video {this_video_path} not exists. Skipping it!')
                            error_num += 1
                            continue

                        video = cv2.VideoCapture(this_video_path)

                        if not video.isOpened():
                            print(f'[warning] Video {this_video_path} open failed. Skipping it!')
                            error_num += 1
                            continue
                        fps = video.get(cv2.CAP_PROP_FPS)
                        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        split_size = self.condition_length + self.generate_length
                        split_num = video_frame_count // split_size

                        for i in range(split_num):
                            temp_dict = {}
                            temp_dict['video_path'] = this_video_path
                            temp_dict['input_video_start_frame'] = i * split_size
                            temp_dict['text'] = text_condition
                            self.data.append(temp_dict)
                        get_video_num += 1
                        progress_bar.update(1)

            elif 'UCF' in opt.dataroot:
                # if opt.phase == 'train':
                #     file = open(os.path.join(opt.dataroot, 'train.txt'))
                #     total_count = len(file.readlines())
                #     file = open(os.path.join(opt.dataroot, 'train.txt'))
                # else:
                #     file = open(os.path.join(opt.dataroot, 'test.txt'))
                #     total_count = len(file.readlines())
                #     file = open(os.path.join(opt.dataroot, 'test.txt'))
                file = open(os.path.join(opt.dataroot, 'tt.txt'))
                total_count = len(file.readlines())
                file = open(os.path.join(opt.dataroot, 'tt.txt'))
                with tqdm(total = total_count) as progress_bar:
                    progress_bar.display('Processing dataset [UCF-101]')
                    for each_file_line in file:
                        each_file_line = each_file_line.split(' ')[0]
                        this_video_path = os.path.join(opt.dataroot, each_file_line)
                        text_condition = each_file_line.split('/')[0]
                        if not os.path.exists(this_video_path):
                            progress_bar.display(f'[warning] Video {this_video_path} not exists. Skipping it!')
                            print(f'[warning] Video {this_video_path} not exists. Skipping it!')
                            error_num += 1
                            continue
                        video = cv2.VideoCapture(this_video_path)
                        if not video.isOpened():
                            progress_bar.write(f'[warning] Video {this_video_path} open failed. Skipping it!')
                            error_num += 1
                            continue
                        fps = video.get(cv2.CAP_PROP_FPS)
                        video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        split_size = self.condition_length + self.generate_length
                        split_num = video_frame_count // split_size
                        if(split_num == 0):
                            progress_bar.write(f'[warning] Skip a short video. No more than {split_size} frame(s)!')
                            continue
                        
                        for i in range(split_num):
                            temp_dict = {}
                            temp_dict['video_path'] = this_video_path
                            temp_dict['input_video_start_frame'] = i * split_size
                            temp_dict['text'] = text_condition
                            for t in range(split_size):
                                succ, _ = video.read()
                            if succ:
                                self.data.append(temp_dict)
                            else:
                                # progress_bar.write(f'[ERROR] Video {this_video_path}, only have no more than {i * split_size} frame, but it annouces there are {video_frame_count} frames!!! TAT')
                                break
                        get_video_num += 1
                        progress_bar.update(1)
            

            elif 'KTH' in opt.dataroot:
                    # if opt.phase == 'train':
                    #     file = open(os.path.join(opt.dataroot, 'train.txt'))
                    #     total_count = len(file.readlines())
                    #     file = open(os.path.join(opt.dataroot, 'train.txt'))
                    # else:
                    #     file = open(os.path.join(opt.dataroot, 'test.txt'))
                    #     total_count = len(file.readlines())
                    #     file = open(os.path.join(opt.dataroot, 'test.txt'))
                    # fileDirs = ['running', 'boxing', 'handclapping', 'handwaving', 'jogging', 'walking']
                    fileDirs = ['handclapping']
                    filelists = []
                    for fileDir in fileDirs:
                        temp = os.listdir(os.path.join(opt.dataroot, fileDir))
                        for i in temp:
                            filelists.append(os.path.join(opt.dataroot, fileDir, i))
                    total_count = len(filelists)
                    with tqdm(total = total_count) as progress_bar:
                        progress_bar.display('Processing dataset [KTH]')
                        for each_file_line in filelists:
                            this_video_path = each_file_line
                            text_condition = each_file_line.split('_')[1]
                            if opt.phase == 'train' and os.path.basename(this_video_path).split('_')[0] not in ['person11', 'person12', 'person13', 'person14', 'person15', 'person16', 'person17', 'person18']:
                                progress_bar.update(1)
                                continue
                            if opt.phase == 'test' and os.path.basename(this_video_path).split('_')[0] not in ['person22', 'person02', 'person03', 'person09', 'person05', 'person06', 'person07', 'person08', 'person10']:
                                progress_bar.update(1)
                                continue
                            if not os.path.exists(this_video_path):
                                progress_bar.display(f'[warning] Video {this_video_path} not exists. Skipping it!')
                                print(f'[warning] Video {this_video_path} not exists. Skipping it!')
                                error_num += 1
                                continue
                            video = cv2.VideoCapture(this_video_path)
                            if not video.isOpened():
                                progress_bar.write(f'[warning] Video {this_video_path} open failed. Skipping it!')
                                error_num += 1
                                continue
                            fps = video.get(cv2.CAP_PROP_FPS)
                            video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            split_size = self.condition_length + self.generate_length
                            split_num = video_frame_count // split_size
                            if(split_num == 0):
                                progress_bar.write(f'[warning] Skip a short video. No more than {split_size} frame(s)!')
                                continue
                            
                            for i in range(split_num):
                                temp_dict = {}
                                temp_dict['video_path'] = this_video_path
                                temp_dict['input_video_start_frame'] = i * split_size
                                temp_dict['text'] = text_condition
                                for t in range(split_size):
                                    succ, _ = video.read()
                                if succ:
                                    self.data.append(temp_dict)
                                else:
                                    # progress_bar.write(f'[ERROR] Video {this_video_path}, only have no more than {i * split_size} frame, but it annouces there are {video_frame_count} frames!!! TAT')
                                    break
                            get_video_num += 1
                            progress_bar.update(1)
            np.save(os.path.join(opt.dataroot, 'data.npy'), np.array(self.data))

            

        if len(self.data) > opt.max_dataset_size:
            self.data = self.data[:opt.max_dataset_size]
        
        self.dataset_size = len(self.data)

        self.batchSize = opt.batchSize

        print(f"[Warning] Get {get_video_num} videos, and get {self.dataset_size} trainable data. {error_num} errors(s).")


    def __getitem__(self, index):
        data_dict = self.data[index]
        video = cv2.VideoCapture(data_dict['video_path'])
        assert video.isOpened() == True
        seg_path_dir, seg_path_floder = os.path.split(data_dict['video_path'])
        seg_video_path = os.path.join(seg_path_dir + '_seg', seg_path_floder)


        input_images = torch.tensor([])
        gen_images = torch.tensor([])
        gen_segs = torch.tensor([]).long()
        gen_temp = torch.tensor([]).long()
        path = data_dict['video_path']

        for i in range(data_dict['input_video_start_frame']):
            video.read()

        for i in range(data_dict['input_video_start_frame'], data_dict['input_video_start_frame'] + self.condition_length):
            success, frame = video.read()
            if frame is None:
                None
            img = self.transform_image(transforms.ToPILImage()(torch.tensor(frame).permute(2, 0, 1)))
            input_images = torch.cat([input_images, img.unsqueeze(0)])
        
        for i in range(data_dict['input_video_start_frame'] + self.condition_length, data_dict['input_video_start_frame'] + self.condition_length + self.generate_length):
            success, frame = video.read()
            img = self.transform_image(transforms.ToPILImage()(torch.tensor(frame).permute(2, 0, 1)))
            gen_images = torch.cat([gen_images, img.unsqueeze(0)])
            # if i - data_dict['input_video_start_frame'] - self.condition_length % self.generate_length == 0:
            #     gen_images = torch.cat([gen_images, gen_temp.unsqueeze(0)])


            img = cv2.imread(os.path.join(seg_video_path, str(i).zfill(5) + '.png'))
            img = torch.tensor(img).long()
            gen_segs = torch.cat([gen_segs, img[:, :, 0].unsqueeze(0)])
           
            

        
        input_images = input_images.permute(1, 0, 2, 3)
        gen_images = gen_images.permute(1, 0, 2, 3)

        input_dict = {
            'path': data_dict['video_path'], 
            'input_images': input_images,
            'gen_images': gen_images,
            'gen_segs': gen_segs,
            'text': data_dict['text'],
        }
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize

    def name(self):
        return 'text_to_video_dataset'