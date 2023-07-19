import torch
import cv2
import numpy as np
from augmentations import generate_hist_augs, opts, mean_domains, std_domains
from histaugan.model import MD_multi
import imageio
from torchvision import transforms
import time

from utils import PPlot


def main():
    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    model = MD_multi(opts)
    model.resume(opts.resume, train=False)
    model.to('cuda:0')
    model.eval()

    # img = imageio.v2.imread(r'D:\jassorRepository\OCELOT_Dataset\what.jpg')
    img = cv2.imread(r'D:\jassorRepository\OCELOT_Dataset\how.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    pplt = PPlot()
    # pplt.add(img)
    img = trnsfrms_val(img).cuda()  # torch.Size([3, 512, 512])
    print(img.shape)
    img1 = img.cpu().numpy()
    print(np.max(img1), np.min(img1))
    z_content = model.enc_c(img.sub(0.5).mul(2).unsqueeze(0))

    # 实验
    for d in range(5):
        for seed in [
            torch.tensor([[10, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 4, 0, 0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 0, 3, 0, 3, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 0, 0, 2, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0, 0, 0, 0, 0, 10]], dtype=torch.float32),
        ]:
            print(seed)
            z_attribute = (seed * std_domains[0] + mean_domains[0]).to('cuda:0')
            domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
            out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
            out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
            # pplt.title(int(seed.sum()))
            pplt.add(out)
    pplt.show()

    # # 实验 -- 确定参数如下所示
    # d = 4
    # for i, j in [(i, j) for i in range(10) for j in range(10)]:
    #     if i >= 6:
    #         pplt.add(np.zeros((10, 10), dtype=np.uint8))
    #         continue
    #     seed = torch.zeros(1, 8, dtype=torch.float32)
    #     if i == 0:
    #         seed[0, 0] = 1 * j
    #     elif i == 1:
    #         seed[0, 1] = 0.1 * j
    #     elif i == 2:
    #         seed[0, 3] = 0.4 * j
    #     elif i == 3:
    #         seed[0, (4, 6)] = 0.3 * j
    #     elif i == 4:
    #         seed[0, 5] = 0.2 * j
    #     elif i == 5:
    #         seed[0, 7] = 1 * j
    #     print(i, j, seed)
    #     z_attribute = (seed * std_domains[0] + mean_domains[0]).to('cuda:0')
    #     domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
    #     out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
    #     out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
    #     # pplt.title(int(seed.sum()))
    #     pplt.add(out)
    # pplt.show()

    # 实验
    # d = 4
    # for i, j in [(i, j) for i in range(10) for j in range(10)]:
    #     if i >= 3:
    #         pplt.add(np.zeros((10, 10), dtype=np.uint8))
    #         continue
    #     seed = torch.zeros(1, 8, dtype=torch.float32)
    #     if i == 0:
    #         seed[0, 7] = 1 * j
    #     elif i == 1:
    #         seed[0, 0] = 1 * j
    #     elif i == 2:
    #         seed[0, (7, 0)] = 1 * j
    #     print(i, j, seed)
    #     z_attribute = (seed * std_domains[0] + mean_domains[0]).to('cuda:0')
    #     domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
    #     out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
    #     out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
    #     # pplt.title(int(seed.sum()))
    #     pplt.add(out)
    # pplt.show()

    # # 实验
    # d = 3
    # for i, j in [(i, j) for i in range(8) for j in range(8)]:
    #     seed = torch.zeros(1, 8, dtype=torch.float32)
    #     seed[0, i] = 1 * j
    #     print(i, j, seed)
    #     z_attribute = (seed * std_domains[0] + mean_domains[0]).to('cuda:0')
    #     domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
    #     out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
    #     out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
    #     # pplt.title(int(seed.sum()))
    #     pplt.add(out)
    # pplt.show()

    # # 控制变量法观察：固定 domain，改变 z_attribute
    # # github code
    # for d in range(5):
    #     for _ in range(5):
    #         seed = torch.randn((1, 8,))
    #         z_attribute = (seed * std_domains[d] + mean_domains[d]).to('cuda:0')
    #         domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
    #         out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
    #         out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
    #         print(out.shape, out.dtype, out.mean())
    #         pplt.add(out)
    # pplt.show()

    # # 控制变量法观察：固定 z_attribute，改变 domain
    # seed = torch.randn((1, 8,))
    # # github code
    # for _ in range(5):
    #     for d in range(5):
    #         z_attribute = (seed * std_domains[d] + mean_domains[d]).to('cuda:0')
    #         domain = torch.eye(5)[d].unsqueeze(0).to('cuda:0')
    #         out = model.gen(z_content, z_attribute, domain).detach().squeeze(0)
    #         out = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
    #         print(out.shape, out.dtype, out.mean())
    #         pplt.add(out)
    # pplt.show()

    # 炳豆方法
    # for iiii in range(1):
    #     aa1 = time.time()
    #     for i in range(5):
    #         out = generate_hist_augs(
    #             img,
    #             domain,
    #             model,
    #             z_content,
    #             same_attribute=False,
    #             new_domain=i,
    #             stats=(mean_domains, std_domains),
    #             device='cuda:0'
    #         )
    #         out_img = out.add(1).div(2).permute(1, 2, 0).cpu().numpy()
    #         # print(np.max(out_img),np.min(out_img))
    #         # imageio.imwrite(rf'D:\jassorRepository\OCELOT_Dataset\{i}.jpg', out_img)
    #     aa2 = time.time()
    #
    #     print(7777, aa2 - aa1)


biao = [
    [],
]

main()
