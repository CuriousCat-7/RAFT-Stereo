import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2 as cv
import os


DEVICE = 'cuda'

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def ptc(img, flo, i, output_dir):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    disp = flo[:,:,0]
    # output point cloud
    h, w = img.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = img
    out_points = points #TODO
    out_colors = colors #TODO
    out_fn = os.path.join(str(output_dir), f'{i}.ply')
    write_ply(out_fn, out_points, out_colors)


def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    sf = lambda x: int(x.split("/")[-1].split(".")[0].split("rect")[-1])
    left_images = sorted(glob.glob(args.left_imgs, recursive=True), key=sf)
    right_images = sorted(glob.glob(args.right_imgs, recursive=True), key = sf)
    print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    with torch.no_grad():
        for i, (imfile1, imfile2) in enumerate(tqdm(list(zip(left_images, right_images)))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            # save
            if args.save_numpy:
                np.save(output_directory / f"{i}.npy", flow_up.cpu().numpy().squeeze())
            if args.save_ply:
                ptc(image1, -flow_up, i, output_directory)

            _, axs = plt.subplots(2,2)
            axs[0][0].imshow(image1[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
            axs[0][1].imshow(image2[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
            axs[1][0].imshow(-flow_up.cpu().numpy().squeeze(), cmap="jet")
            plt.savefig(output_directory / f"{i}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--save_ply', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    demo(args)
