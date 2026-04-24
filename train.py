import os
import torch
import random
import shutil
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.TSANet import TSANet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import math
import torch.nn.functional as F

class AugmentationModule(nn.Module):
    def __init__(self):
        super(AugmentationModule, self).__init__()
        self.crop_size = 128
        
    def weak_aug(self, x):
        n, c, h, w = x.shape
        if h <= self.crop_size or w <= self.crop_size:
            return x
        
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        return x[:, :, top:top+self.crop_size, left:left+self.crop_size]

    def strong_aug(self, x):
        down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # 2. 上采样 (Upsample) 2.0x 恢复原尺寸
        up = F.interpolate(down, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        return up
        
    def forward(self, x):
        weak_output = self.weak_aug(x)
        strong_output = self.strong_aug(weak_output)
        
        return weak_output, strong_output

def calculate_beta(initial_alpha, current_step, total_steps):
    if total_steps == 0:
        return initial_alpha
    cos_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
    return initial_alpha * cos_decay

opt = option().parse_args()

def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init(seed=42):
    seed_torch(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if opt.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found")

# =========================================================
# 训练循环
# =========================================================
def train(epoch, global_step, total_training_steps, aug_opt, msr_loss_fn):
    model.train()
    loss_print = 0
    pic_cnt = 0
    
    # torch.autograd.set_detect_anomaly(opt.grad_detect)

    pbar = tqdm(training_data_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        im1, im2 = batch[0].cuda(), batch[1].cuda()
        
        # 主模型前向
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)  
        else:
            output_rgb = model(im1)  
            
        gt_rgb = im2
        
        # --- 计算外部损失 (External Loss) ---
        loss_hvi = 0
        if hasattr(model, 'HVIT'):
            output_hvi = model.HVIT(output_rgb)
            gt_hvi = model.HVIT(gt_rgb)
            loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
            
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        external_loss = loss_rgb + opt.HVI_weight * loss_hvi
        
        # =========================================================================
        # 核心：内部自监督损失 (Internal Scale Consistency Loss)
        # =========================================================================

        weak_output, strong_output = aug_opt(output_rgb)
        internal_loss = msr_loss_fn(weak_output, strong_output)
        initial_alpha = 0.05 
        beta = calculate_beta(initial_alpha, global_step, total_training_steps)
        
        # 4. 总损失
        loss = external_loss + beta * internal_loss
        
        # =========================================================================
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print += loss.item()
        pic_cnt += 1
        global_step += 1

        pbar.set_postfix(loss=loss.item(), ext=external_loss.item(), int=internal_loss.item())

    avg_loss = loss_print / pic_cnt if pic_cnt > 0 else 0
    print(f"\n===> Epoch[{epoch}] 完成: 平均损失: {avg_loss:.4f}")

    # 保存训练图
    with torch.no_grad():
        if not os.path.exists(opt.val_folder+'training'):          
            os.makedirs(opt.val_folder+'training') 
        
        save_img = torch.cat([weak_output[0], strong_output[0]], dim=2) 
        transforms.ToPILImage()(save_img.clip(0,1).cpu()).save(opt.val_folder+'training/ssl_compare.png')

    return loss_print, pic_cnt, global_step


def save_best_checkpoint(model_state, model_path):
    if not os.path.exists("./weights"): os.mkdir("./weights") 
    if not os.path.exists("./weights/best"): os.mkdir("./weights/best")
    best_model_out_path = os.path.join(model_path, "best_model.pth")
    torch.save(model_state, best_model_out_path)
    print(f"======> Best model saved to {best_model_out_path}")

def checkpoint(epoch):
    if not os.path.exists("./weights"): os.mkdir("./weights") 
    if not os.path.exists("./weights/train"): os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    train_set = None
    test_set = None
    if opt.lolv1:
        train_set = get_lol_training_set(opt.data_train_lolv1, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv1)
    else:
        try:
            train_set = get_lol_training_set(opt.data_train_lolv1, size=opt.cropSize)
            test_set = get_eval_set(opt.data_val_lolv1)
        except: pass
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = TSANet()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        if os.path.exists(pth):
            model.load_state_dict(torch.load(pth, map_location="cpu"))
    model = model.cuda() 
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=1e-7)
    return optimizer,scheduler

def init_loss():
    L1_loss= L1Loss(loss_weight=opt.L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=opt.D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=opt.E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = 1.0 ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    train_init(seed=42)
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    aug_opt = AugmentationModule().cuda()
    msr_loss_fn = nn.MSELoss().cuda()
    
    total_training_steps = opt.nEpochs * len(training_data_loader)
    global_step = 0
    if opt.start_epoch > 0:
        global_step = opt.start_epoch * len(training_data_loader)
        
    psnr, ssim, lpips = [], [], []
    best_psnr = 0.0
    start_epoch = 0
    if opt.start_epoch > 0: start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder): os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        loss_print, pic_num, global_step = train(epoch, global_step, total_training_steps, aug_opt, msr_loss_fn)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            
            output_folder = 'val_results/' 
            if opt.lolv1: output_folder = 'LOLv1/'
            im_dir = opt.val_folder + output_folder + '*.png'
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, norm_size=True, LOL=opt.lolv1, v2=opt.lolv2real, alpha=0.8)
            
            label_dir = opt.data_valgt_lolv1 if opt.lolv1 else ''
            if label_dir and os.path.exists(label_dir):
                avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
                print(f"===> Epoch[{epoch}] PSNR: {avg_psnr:.4f} (Best: {max(best_psnr, avg_psnr):.4f})")
                
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    save_best_checkpoint(model.state_dict(), "./weights/best")
                    dest = os.path.join(opt.val_folder, "best_visual_results")
                    if os.path.exists(dest): shutil.rmtree(dest)
                    shutil.copytree(opt.val_folder + output_folder, dest)
                psnr.append(avg_psnr)
            torch.cuda.empty_cache()
            
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("| Epochs | PSNR |\n|---|---|\n")
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | {psnr[i]:.4f} |\n")