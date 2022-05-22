import cv2
import glob
import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings

sys.path.append('.')

from models.autoencoder import *
from models.util import *
from util.test_parser import test_parser
from test_helper import get_relative_pose

"""
Example usage (note the quotation marks):
`python data/clean_dataset.py ../../dataset/test ../../re10k.ckpt`
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def image_loader(path, input_size):
    image = cv2.imread(path)
    h, w, _ = image.shape
    input_h, input_w = input_size, input_size
    image = cv2.resize(image, (input_w, input_h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.unsqueeze(transforms.ToTensor()(image), 0), h, w

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("need to input files to clean and model")
        quit()

    args = SimpleNamespace(scale_rotate=0.01, scale_translate=0.01, padding_mode='zeros')
    dataset = sys.argv[1]
    video_paths = sorted(os.listdir(dataset))
    print(f"#Videos: {len(video_paths)}")

    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    rotate = Rotate(args)
    decoder = Decoder(args)

    encoder_3d = nn.DataParallel(encoder_3d).to(device)
    encoder_traj = nn.DataParallel(encoder_traj).to(device)
    rotate = nn.DataParallel(rotate).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    checkpoint = torch.load(sys.argv[2], map_location=torch.device(device))
    encoder_3d.load_state_dict(checkpoint['encoder_3d'])
    encoder_traj.load_state_dict(checkpoint['encoder_traj'])
    decoder.load_state_dict(checkpoint['decoder'])
    rotate.load_state_dict(checkpoint['rotate'])

    encoder_3d.eval()
    encoder_traj.eval()
    decoder.eval()
    rotate.eval()

    to_image = transforms.ToPILImage()
    i = 0
    norms_fro = []
    norms_nuc = []
    for video_path in video_paths:
        vid_dir = os.path.join(dataset, video_path)
        if not os.path.isdir(vid_dir):
            continue
        frames = sorted(os.listdir(vid_dir))
        first_vid = frames[0]
        second_vid = frames[0]
        first_path = os.path.join(vid_dir, first_vid)
        second_path = os.path.join(vid_dir, second_vid)
        im, h, w = image_loader(first_path, 256)
        im1, _, _ = image_loader(second_path, 256)
        scene_rep = encoder_3d(im)

        # imgs_pair = torch.cat([im, im1], dim=1)  # t x 6 x h x w
        # poses = encoder_traj(imgs_pair)
        # z = euler2mat(poses)
        z = get_relative_pose(encoder_traj, torch.squeeze(im), torch.squeeze(im1))

        rot_codes = rotate(scene_rep, z)
        reconstruct_im = decoder(rot_codes)
        reconstruct_im = torch.clamp(reconstruct_im, 0, 1)

        im = torch.squeeze(im)[:, :70, :70]
        reconstruct_im = torch.squeeze(reconstruct_im)[:, :70, :70]

        n_fro = torch.sqrt(torch.sum(torch.square(im-reconstruct_im))).item()
        norms_fro.append(n_fro)
        n_nuc = torch.sum(torch.abs(im-reconstruct_im)).item()
        norms_nuc.append(n_nuc)
        print(n_fro, n_nuc)

        print(im.shape)
        orig_im = to_image(im)
        pil_im = to_image(reconstruct_im)
        if n_fro >= 50:
            pil_im.show()
        # i += 1
        # if i >= 30:
        #     break
    plt.figure(0)
    plt.hist(norms_nuc, density=False, bins=30, label='nuc')
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()

    plt.figure(1)
    plt.hist(norms_fro, density=False, bins=30, label='fro')
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()
