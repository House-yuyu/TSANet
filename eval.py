import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.TSANet import TSANet


def eval(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    
    # 设置门的逻辑
    if LOL: model.trans.gated = True
    elif v2: model.trans.gated2 = True; model.trans.alpha = alpha
    elif unpaired: model.trans.gated2 = True; model.trans.alpha = alpha
    
    if not os.path.exists(output_folder):          
        os.makedirs(output_folder, exist_ok=True)
            
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            output = model(input**gamma) 
            
        output = torch.clamp(output.cuda(), 0, 1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(os.path.join(output_folder, name[0]))
        torch.cuda.empty_cache()
        
    print('===&gt; End evaluation')
    # 恢复门的状态
    if LOL: model.trans.gated = False
    elif v2: model.trans.gated2 = False
    torch.set_grad_enabled(True)

if __name__ == '__main__':
    
    eval_parser = argparse.ArgumentParser(description='Evaluation Script')
    
    # 为独立运行 eval.py 定义它自己的参数
    eval_parser.add_argument('--model_path', type=str, required=True, help='/home/zhappy-701/Desktop/wu/HVI-CIDNet-master-now/weights/best/best_model.pth')
    eval_parser.add_argument('--input_dir', type=str, required=True, help='/home/zhappy-701/Desktop/wu/LOLdataset/eval15/low')
    eval_parser.add_argument('--output_dir', type=str, default='./output/default/', help='Path to save the enhanced images')
    
    # 添加一些模型行为控制参数
    eval_parser.add_argument('--is_lolv1', action='store_true', help='Set this if the model is for LOL-v1')
    eval_parser.add_argument('--is_lolv2', action='store_true', help='Set this if the model is for LOL-v2')
    eval_parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for LOL-v2 models')
    eval_parser.add_argument('--gamma', type=float, default=1.0, help='Gamma correction value')

    ep = eval_parser.parse_args()

    # --- [独立运行的逻辑] ---
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    # 创建模型
    eval_net = CIDNet().cuda()
    

    eval_dataset = get_eval_set(ep.input_dir)
    eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=1, batch_size=1, shuffle=False)
    
    # 调用核心的 eval 函数
    eval(model=eval_net, 
         testing_data_loader=eval_data_loader, 
         model_path=ep.model_path, 
         output_folder=ep.output_dir, 
         norm_size=True,  # 假设默认是 True
         LOL=ep.is_lolv1, 
         v2=ep.is_lolv2, 
         alpha=ep.alpha, 
         gamma=ep.gamma)