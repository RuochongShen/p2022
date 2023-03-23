import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.diffusion import *
from models.UNet import *
from models.WideResNet import *
from models.blocks import *

from train import *
from datasets import testData_v2
from pdf_generator.pdf_result_generator import *


def my_training1(device, train_dl_path, test_dl_path, store_path="ddpm_mytrain1.pt", n_chosen=1, cond_ch=1, repaint=True,
                 miuWater=0.19, loadfile='testset.txt', totaln=None, batch_size=4, n_steps=1000, min_beta=10 ** -4,
                 max_beta=0.02, MyBlock=MyBlock1, test_save_folder='train_sampling', sinogram=False):
    dataset = testData_v2(path=train_dl_path, istest=False, n_chosen=n_chosen, cond_ch=cond_ch, miuWater=miuWater,
                          loadfile=loadfile, totaln=totaln, sinogram=sinogram)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(loader))

    ddpm = MyDDPM(MyUNet(n_steps, cond_ch=cond_ch, MyBlock=MyBlock).to(device), n_steps=n_steps, min_beta=min_beta,
                  max_beta=max_beta, device=device)
    training_loop(ddpm, loader, n_epochs=10, optim=Adam(ddpm.parameters(), lr=10 ** -4), device=device,
                  store_path=store_path)

    best_model = MyDDPM(MyUNet(n_steps, cond_ch=cond_ch, MyBlock=MyBlock).to(device), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("loaded, testing")

    testset = testData_v2(path=test_dl_path, istest=True, n_chosen=n_chosen, cond_ch=cond_ch, miuWater=miuWater,
                          loadfile=loadfile, totaln=totaln, sinogram=sinogram)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    n_test = len(test_loader)
    print(n_test)
    rmse, ssim, psnr = 0, 0, 0
    rmse_mat, ssim_mat, psnr_mat = 0, 0, 0
    num = 0
    with torch.no_grad():
        for i, (mt, gt) in tqdm(enumerate(test_loader)):
            # print((mt.shape,gt.shape))
            print(f"{i} start")
            mt = mt.to(device)
            gt = gt.to(device)
            if repaint:
                marTest = generate_new_images(best_model, mt, im_gt=gt, n_samples=1, device=device, folder_name=test_save_folder+str(i))
            else:
                marTest = generate_new_images(best_model, mt, im_gt=gt, n_samples=1, device=device, folder_name=test_save_folder + str(i), threshold_m=-np.Inf)
            im1 = (gt[0, 0, :, :].cpu().numpy()) # / 1000 * 0.19 + 0.19
            im2 = marTest[0, :, :].cpu().numpy()

            rmse_n, psnr_n, ssim_n = calc_met_rsp(im1, im2, in_matlab=False)
            rmse_mat_n, psnr_mat_n, ssim_mat_n = calc_met_rsp(im1, im2, in_matlab=True)
            rmse += rmse_n
            psnr += psnr_n
            ssim += ssim_n
            rmse_mat += rmse_mat_n
            psnr_mat += psnr_mat_n
            ssim_mat += ssim_mat_n

            num += 1
            print(f"Average of metrics are: rmse: {rmse/num}; ssim: {ssim/num};  psnr: {psnr/num}")  # {ssim2/num};
            print(f"Average of metrics (MATLAB) are: rmse: {rmse_mat / num}; ssim: {ssim_mat / num};  psnr: {psnr_mat / num}")

    print(f"Average of metrics are: rmse: {rmse/n_test}; ssim: {ssim/n_test}; psnr: {psnr/n_test}")
    print(f"Average of metrics (MATLAB) are: rmse: {rmse_mat/n_test}; ssim: {ssim_mat/n_test};  psnr: {psnr_mat/n_test}")


def load_model(device, store_path="ddpm_mytrain1.pt", n_steps=1000):
    best_model = MyDDPM(MyUNet().to(device), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    return best_model


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:256'
    store_path = "ddpm_mytrain1.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))
    mode = "baseline"

    if mode == "baseline":
        my_training1(device, 'D:\\pycharm projects\\adn\\data\\deep_lesion\\train\\',
                 'D:\\pycharm projects\\adn\\data\\deep_lesion\\test\\', store_path='ddpm_baseline1.pt',repaint=False,
                 n_chosen=1, cond_ch=1, miuWater=0.19, loadfile='testset.txt', totaln=None, batch_size=4, n_steps=1000,
                 min_beta=10 ** -4, max_beta=0.02)

    if mode == "image repaint":
        my_training1(device, 'D:\\pycharm projects\\adn\\data\\deep_lesion\\train\\',
                     'D:\\pycharm projects\\adn\\data\\deep_lesion\\test\\', store_path='ddpm_repaint1.pt',
                     n_chosen=1, cond_ch=1, miuWater=0.19, loadfile='testset.txt', totaln=None, batch_size=4,
                     n_steps=1000, repaint=True, min_beta=10 ** -4, max_beta=0.02)

    if mode == "sino repaint":
        my_training1(device, 'D:\\pycharm projects\\adn\\data\\deep_lesion\\train\\',
                     'D:\\pycharm projects\\adn\\data\\deep_lesion\\test\\', store_path='ddpm_repaint1.pt',
                     n_chosen=1, cond_ch=1, miuWater=0.19, loadfile='testset.txt', totaln=None, batch_size=4,
                     n_steps=1000, repaint=True, sinogram=True, min_beta=10 ** -4, max_beta=0.02)

    # show_first_batch(loader)
    # show_forward(ddpm, loader, device)
