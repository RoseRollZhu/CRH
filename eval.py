import argparse
import os
import numpy as np
import cv2
import torch

import lpips
from pytorch_msssim import ms_ssim
from tqdm import tqdm

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cal_ssim(path1, path2):
    list1dir = os.listdir(path1)
    list2dir = os.listdir(path2)
    file1list = []
    file2list = []
    ssim_dis = []
    for i in list1dir:
        temp = os.listdir(os.path.join(path1, i))
        for j in temp:
            file1list.append(os.path.join(path1, i, j))
    
    for i in list2dir:
        temp = os.listdir(os.path.join(path2, i))
        for j in temp:
            file2list.append(os.path.join(path2, i, j))

    for i in tqdm(range(min(len(file2list), len(file1list)))):
        try:
            img0 = cv2.imread(file1list[i])
            img1 = cv2.imread(file2list[i])
            dist = ssim(img0, img1)
        except:
            print('[warning] skip a photo')
            continue
        ssim_dis.append(dist)
    ssim_dis_mean = np.mean(ssim_dis)
    score_ssim = np.sqrt( 1 - 2 * (np.min([np.max([0.2, ssim_dis_mean]), 0.7]) - 0.2) )
    return score_ssim


def cal_lpips(path1, path2):
    list1dir = os.listdir(path1)
    list2dir = os.listdir(path2)
    file1list = []
    file2list = []
    lpips_dis = []
    model = lpips.LPIPS(net='vgg').cuda()
    for i in list1dir:
        temp = os.listdir(os.path.join(path1, i))
        for j in temp:
            file1list.append(os.path.join(path1, i, j))
    
    for i in list2dir:
        temp = os.listdir(os.path.join(path2, i))
        for j in temp:
            file2list.append(os.path.join(path2, i, j))

    for i in tqdm(range(min(len(file2list), len(file1list)))):
        try:
            img0 = lpips.im2tensor(lpips.load_image(file1list[i])).cuda() # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(file2list[i])).cuda()
        except:
            print('[warning] skip a photo')
            continue
        
        dist = model(img0, img1)
        lpips_dis.append(dist.cpu().detach().numpy().max())
    lpips_dis_mean = np.mean(lpips_dis)
    score_lpips = np.sqrt( 1 - 2 * (np.min([np.max([0.2, lpips_dis_mean]), 0.7]) - 0.2) )
    return score_lpips



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='1', help='The ID of GPU to use.')
    parser.add_argument('--batch_size', type=int, default=20, help='num img of a batch')

    parser.add_argument('--ori_path', type=str, default='results/test_samples/KTH-4-2-5-handclapping/real', help='Input directory with images.')

    parser.add_argument('--adv_path', type=str, default='results/test_samples/KTH-4-2-5-handclapping/synthesized', help='adv directory with images.')


    # parser.add_argument('--fid', type=bool, default=False)
    # parser.add_argument('--lpips', type=bool, default=False)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    score_lpips = cal_lpips(opt.adv_path, opt.ori_path)
    print('score_lpips: ', score_lpips)
    score_ssim = cal_ssim(opt.adv_path, opt.ori_path)
    print('score_ssim: ', score_ssim)
