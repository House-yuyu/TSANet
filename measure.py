import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform
import re


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr
    
# metrics 函数也保持原样
def metrics(im_dir, label_dir, use_GT_mean):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    # 路径分隔符处理优化
    gt_files = {re.findall(r'\d+', os.path.basename(f))[0]: f for f in glob.glob(os.path.join(label_dir, '*'))}

    im_list = sorted(glob.glob(im_dir.replace(os.path.sep, '/')))
    for item in tqdm(im_list):
        im1 = Image.open(item).convert('RGB')

        base_name = os.path.basename(item)
        match = re.findall(r'\d+', base_name)
        
        if not match:
            print(f"Warning: Could not find number in filename {base_name}. Skipping.")
            continue
        
        core_number = match[0]
        
        # 在 GT 文件字典中查找对应的文件
        if core_number not in gt_files:
            print(f"Warning: No matching GT file found for {base_name} (core number: {core_number}). Skipping.")
            continue

        gt_path = gt_files[core_number]
        im2 = Image.open(gt_path).convert('RGB')
        
        n += 1 # 只有在成功找到匹配时才增加计数
        
        (h, w) = im2.size
        im1 = im1.resize((h, w))
        im1 = np.array(im1)
        im2 = np.array(im2)
        
        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)
        
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)
        ex_p0 = lpips.im2tensor(im1).cuda()
        ex_ref = lpips.im2tensor(im2).cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()
        torch.cuda.empty_cache()
    
    avg_psnr = avg_psnr / n if n != 0 else 0
    avg_ssim = avg_ssim / n if n != 0 else 0
    avg_lpips = avg_lpips / n if n != 0 else 0
    return avg_psnr, avg_ssim, avg_lpips

if __name__ == '__main__':
    
    mea_parser = argparse.ArgumentParser(description='Measure Script')
    mea_parser.add_argument('--im_dir', type=str, required=True, help='/media/zh_701/新加卷/WYY/HVI-CIDNet-masterzuiniu/results/best_visual_results/*.png')
    # mea_parser.add_argument('--im_dir', type=str, required=True, help='/home/zhappy-701/Desktop/wu/HVI-CIDNet-master/HVI-CIDNet-master/results/best_visual_results/*.png')
    mea_parser.add_argument('--label_dir', type=str, required=True, help="/media/zh_701/新加卷/WYY/datasets/LOLdataset/eval15/high")
    mea_parser.add_argument('--use_GT_mean', action='store_true', help="/media/zh_701/新加卷/WYY/datasets/LOLdataset/eval15/high")
    
    mea = mea_parser.parse_args()

    # 直接调用上面的 metrics 函数
    avg_psnr, avg_ssim, avg_lpips = metrics(mea.im_dir, mea.label_dir, mea.use_GT_mean)
    
    print("===&gt; Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===&gt; Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===&gt; Avg.LPIPS: {:.4f} ".format(avg_lpips))