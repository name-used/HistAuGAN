import os

import torch

from histaugan.datasets import dataset_single
from histaugan.model import MD_multi
from histaugan.options import TestOptions
from histaugan.saver import save_imgs, save_concat_imgs


def main():
    # opts

    import argparse

    parser = argparse.ArgumentParser()

    # # data loader related
    # parser.add_argument(
    #     '--dataroot', type=str, required=True, help='path of data')
    parser.add_argument('--num_domains', type=int, default=5)
    # parser.add_argument(
    #     '--phase', type=str, default='train', help='phase for dataloading')
    # parser.add_argument(
    #     '--batch_size', type=int, default=2, help='batch size')
    # parser.add_argument(
    #     '--resize_size', type=int, default=256, help='resized image size for training')
    parser.add_argument(
        '--crop_size', type=int, default=216, help='cropped image size for training')
    parser.add_argument(
        '--input_dim', type=int, default=3, help='# of input channels for domain A')
    # parser.add_argument(
    #     '--nThreads', type=int, default=8, help='# of threads for data loader')
    # parser.add_argument(
    #     '--no_flip', action='store_true', help='specified if no flipping')

    # # ouptput related
    # parser.add_argument(
    #     '--name', type=str, default='trial', help='folder name to save outputs')
    # parser.add_argument(
    #     '--display_dir', type=str, default='./logs', help='path for saving display results')
    # parser.add_argument('--result_dir', type=str, default='./results',
    #                          help='path for saving result images and models')
    # parser.add_argument(
    #     '--display_freq', type=int, default=10, help='freq (iteration) of display')
    # parser.add_argument(
    #     '--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    # parser.add_argument(
    #     '--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    # parser.add_argument(
    #     '--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    parser.add_argument('--concat', type=int, default=1,
                             help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    parser.add_argument(
        '--dis_scale', type=int, default=3, help='scale of discriminator')
    parser.add_argument('--dis_norm', type=str, default='None',
                             help='normalization layer in discriminator [None, Instance]')
    parser.add_argument('--dis_spectral_norm', action='store_true',
                             help='use spectral normalization in discriminator')
    parser.add_argument(
        '--lr_policy', type=str, default='step', help='type of learn rate decay')  # MDMM used lambda
    parser.add_argument(
        '--n_ep', type=int, default=1200, help='number of epochs')  # 400 * d_iter
    parser.add_argument('--n_ep_decay', type=int, default=600,
                             help='epoch start decay learning rate, set -1 if no decay')  # 200 * d_iter
    parser.add_argument('--resume', type=str, default=None,
                             help='specified the dir of saved models for resume the training')
    parser.add_argument('--d_iter', type=int, default=3,
                             help='# of iterations for updating content discriminator')
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_cls_G', type=float, default=5.0)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    opts = parser.parse_args()

    # model
    print('\n--- load model ---')
    model = MD_multi(opts=opts)
    model.setgpu(0)
    model.resume(model_dir=r'D:\jassor\workspaceDeepLearning\HistAuGAN\gan_weights.pth', train=False)
    model.eval()

    # test
    print('\n--- testing ---')
    for d in range(opts.num_domains):
        # image 即 torch[b, c, h, w] -> torch.float32(cuda) 输入
        # c_org 则是该 image 的域独热编码
        image = torch.zeros(1, 3, 512, 512, dtype=torch.float32, device='cuda:0')
        for _ in range(5):
            with torch.no_grad():
                outputs = model.test_forward_random(image)
                print(outputs.shape)
    return


if __name__ == '__main__':
    main()
