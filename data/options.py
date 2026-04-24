import argparse

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, &gt;0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # --- [ Paths for Training Datasets ] ---
    parser.add_argument('--data_train_lolblur', type=str, default='./datasets/LOL_blur/train', help='Path to LOL-blur training data')
    parser.add_argument('--data_train_lolv1', type=str, default='/home/bjtc/wyy/LOLdataset/Train/', help='Path to LOL-v1 training data')
    parser.add_argument('--data_train_lolv2real', type=str, default='/media/zh_701/新加卷/WYY/LOL_v2/LOL_v2/Real_captured/Train/', help='Path to LOL-v2 Real training data')
    parser.add_argument('--data_train_lolv2syn', type=str, default='./datasets/LOLv2/Synthetic/Train', help='Path to LOL-v2 Synthetic training data')
    parser.add_argument('--data_train_sid', type=str, default='/media/zh_701/新加卷/WYY/datasets/Sony_total_dark/train/', help='Path to SID training data')
    parser.add_argument('--data_train_sice', type=str, default='./datasets/SICE/Dataset/train', help='Path to SICE training data')
    parser.add_argument('--data_train_fivek', type=str, default='./datasets/FiveK/train', help='Path to FiveK training data')

    # --- [ Paths for Validation Input ] ---
    parser.add_argument('--data_val_lolblur', type=str, default='./datasets/LOL_blur/eval/low_blur', help='Path to LOL-blur validation input')
    parser.add_argument('--data_val_lolv1', type=str, default='/home/bjtc/wyy/LOLdataset/Test/low', help='Path to LOL-v1 validation input')
    parser.add_argument('--data_val_lolv2real', type=str, default='/media/zh_701/新加卷/WYY/LOL_v2/LOL_v2/Real_captured/Test/Low', help='Path to LOL-v2 Real validation input')
    parser.add_argument('--data_val_lolv2syn', type=str, default='./datasets/LOLv2/Synthetic/Test/Low', help='Path to LOL-v2 Synthetic validation input')
    parser.add_argument('--data_val_sid', type=str, default='/media/zh_701/新加卷/WYY/datasets/Sony_total_dark/eval/short', help='Path to SID validation input')
    parser.add_argument('--data_val_sicemix', type=str, default='./datasets/SICE/Dataset/eval/test', help='Path to SICE-mix validation input')
    parser.add_argument('--data_val_sicegrad', type=str, default='./datasets/SICE/Dataset/eval/test', help='Path to SICE-grad validation input')
    parser.add_argument('--data_val_fivek', type=str, default='./datasets/FiveK/test/input', help='Path to FiveK validation input') # Corrected from data_test_fivek

    # --- [ Paths for Validation Groundtruth ] ---
    parser.add_argument('--data_valgt_lolblur', type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/', help='Path to LOL-blur validation GT')
    parser.add_argument('--data_valgt_lolv1', type=str, default='"/home/bjtc/wyy/LOLdataset/Test/high', help='Path to LOL-v1 validation GT')
    parser.add_argument('--data_valgt_lolv2real', type=str, default='/media/zh_701/新加卷/WYY/LOL_v2/LOL_v2/Real_captured/Test/Normal', help='Path to LOL-v2 Real validation GT')
    parser.add_argument('--data_valgt_lolv2syn', type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/', help='Path to LOL-v2 Synthetic validation GT')
    parser.add_argument('--data_valgt_sid', type=str, default='/media/zh_701/新加卷/WYY/datasets/Sony_total_dark/eval/long/', help='Path to SID validation GT')
    parser.add_argument('--data_valgt_sicemix', type=str, default='./datasets/SICE/Dataset/eval/target/', help='Path to SICE-mix validation GT')
    parser.add_argument('--data_valgt_sicegrad', type=str, default='./datasets/SICE/Dataset/eval/target/', help='Path to SICE-grad validation GT')
    parser.add_argument('--data_valgt_fivek', type=str, default='./datasets/FiveK/test/target/', help='Path to FiveK validation GT')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # --- [ Loss Weights ] ---
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    
    # --- [ Augmentation and Training Tricks ] ---
    parser.add_argument('--gamma', type=bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    # --- [ Dataset Selection ] ---
    # Choose which dataset you want to train. Please only provide one flag.
    parser.add_argument('--lolv1', action='store_true', help='Use LOL-v1 dataset')
    parser.add_argument('--lolv2real', action='store_true', help='Use LOL-v2 Real dataset')
    parser.add_argument('--lolv2syn', action='store_true', help='Use LOL-v2 Synthetic dataset')
    parser.add_argument('--lolblur', action='store_true', help='Use LOL-blur dataset')
    parser.add_argument('--sid', action='store_true', help='Use SID dataset')
    parser.add_argument('--sicemix', action='store_true', help='Use SICE-mix dataset')
    parser.add_argument('--sicegrad', action='store_true', help='Use SICE-grad dataset')
    parser.add_argument('--fivek', action='store_true', help='Use FiveK dataset')
    
    return parser