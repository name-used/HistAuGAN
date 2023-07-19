import json

import torch
import cv2
import numpy as np
from augmentations import generate_hist_augs, opts, mean_domains, std_domains
from histaugan.model import MD_multi
import os
import imageio
from torchvision import transforms
import time

from utils import PPlot


def main():
    # 基本信息
    image_root = r'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image'
    trans_root = r'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image_transforms'
    meta_path = r'D:\jassorRepository\OCELOT_Dataset\jassor\tissue\image_transforms\meta.json'

    # 写配置信息
    os.makedirs(trans_root, exist_ok=True)
    with open(meta_path, 'w+') as f:
        json.dump({
            'message': [
                'd(0, 1, 3)-Za(0, 5)',
                'd(0, 1, 3)-Za(1, 2, 3, 4)',
                'd(2, 4)-Za(0, 5)',
                'd(2, 4)-Za(1, 3)',
                'd(2, 4)-Za(2, 4)',
            ],
            'group': {
                f'g{i}': [(d, za) for d in ds for za in zas]
                for i, (ds, zas) in enumerate([
                    [(0, 1, 3), (0, 5)],
                    [(0, 1, 3), (1, 2, 3, 4)],
                    [(2, 4), (0, 5)],
                    [(2, 4), (1, 3)],
                    [(2, 4), (2, 4)],
                ])
            }
        }, f, indent=4)

    # 集组信息
    SEEDS = [
        torch.tensor([[10, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        torch.tensor([[0, 0, 0, 4, 0, 0, 0, 0]], dtype=torch.float32),
        torch.tensor([[0, 0, 0, 0, 3, 0, 3, 0]], dtype=torch.float32),
        torch.tensor([[0, 0, 0, 0, 0, 2, 0, 0]], dtype=torch.float32),
        torch.tensor([[0, 0, 0, 0, 0, 0, 0, 10]], dtype=torch.float32),
    ]

    # 加载模型和权重
    model = MD_multi(opts)
    model.resume(opts.resume, train=False)
    model.to('cuda:0')
    model.eval()

    # 读取图片
    for index in range(1, 401):
        code = str(index).zfill(3)
        image = cv2.imread(rf'{image_root}\{code}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 按他的方法标准化
        image = (image / 255 - 0.5) * 2
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        print(f'start {code} with shape {image.shape} dtype {image.dtype} and max {image.max()} min {image.min()} mean {image.mean()}')

        # d 从 0 - 4 编号在前
        # Za 从 0 - 5 编号在后， Za 信息见 SEEDS
        for d in range(5):
            for j, seed in enumerate(SEEDS):
                # 拆图
                temp = np.zeros(shape=(1024, 1024, 3), dtype=np.float32)
                s = 512
                k = 512
                for x, y in [(x, y) for x in range(0, 1024, s) for y in range(0, 1024, s)]:
                    # 生成 Zc
                    z_content = model.enc_c(torch.Tensor(image[:, :, y: y+k, x: x+k]).cuda())

                    # Za 和 d 各自标准化
                    domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
                    z_attribute = (seed * std_domains[0] + mean_domains[0]).to('cuda:0')
                    # 生成结果
                    out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
                    out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
                    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    temp[y: y+k, x: x+k, :] += out
                # 保存
                if not os.path.exists(rf'{trans_root}\{d}_{j}'):
                    os.makedirs(rf'{trans_root}\{d}_{j}', exist_ok=True)
                print(f'\t ->  {code} with shape {temp.shape} dtype {temp.dtype} and max {np.max(temp)} min {np.min(temp)} mean {temp.mean()}')
                cv2.imwrite(rf'{trans_root}\{d}_{j}\{code}.jpg', (temp * 255).astype(np.uint8))


main()
