import cv2
import glob
import sys
import os
import shutil
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
show_plots = False

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
    if dataset[-1] == '/':
        dataset = dataset[:-1]
    video_paths = sorted(os.listdir(dataset))
    print(f"#Videos: {len(video_paths)}")

    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    rotate = Rotate(args)
    decoder = Decoder(args)

    flow = Flow(args)

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
    for video_path in video_paths:
        vid_dir = os.path.join(dataset, video_path)
        if not os.path.isdir(vid_dir):
            continue

        with torch.no_grad():
            frames = sorted(os.listdir(vid_dir))
            first_vid = frames[0]
            first_path = os.path.join(vid_dir, first_vid)
            im, h, w = image_loader(first_path, 256)
            im = im.to(device)
            scene_rep = encoder_3d(im)
        # z = get_relative_pose(encoder_traj, torch.squeeze(im), torch.squeeze(im))
        rot_codes = rotate.module.second_part(scene_rep)
        reconstruct_im = decoder(rot_codes)
        full_reconstruct_im = torch.squeeze(torch.clamp(reconstruct_im, 0, 1))

        pil_im = to_image(full_reconstruct_im)
        print(scene_rep.shape)
        histo, _ = scene_rep.abs().max(1)
        print(histo.shape)
        print(histo.mean(), histo.max(), histo.median())

        histo_rep = histo.unsqueeze(1).repeat(1, 32, 1, 1, 1)
        histo_scale = 0.5
        scene_rep2 = scene_rep * (histo_scale) #torch.logical_and(histo_rep > 0.005, histo_rep < 0.05)

        # flow_rep = torch.cat([torch.zeros(1, 32, 64, 64, 1), torch.zeros(1, 32, 64, 64, 1), torch.zeros(1, 32, 64, 64, 1)], dim=4) / 10
        # flow_rep[:, 0:32, 10:20, 0:64, 1] = -1/8

        # scene_rep2 = flow(scene_rep2, flow_rep)
        rot_codes2 = rotate.module.second_part(scene_rep2)
        reconstruct_im2 = decoder(rot_codes2)
        full_reconstruct_im2 = torch.squeeze(torch.clamp(reconstruct_im2, 0, 1))
        pil_im2 = to_image(full_reconstruct_im2)
        # pil_im.show()
        pil_im2.show()

        print(torch.mean((histo_rep <= 0.0025).float()))


        # histo = true_mask.flatten().detach().numpy()
        # plt.figure(0).clear()
        # plt.hist(histo, density=False, bins=100, label='nuc')
        # plt.ylabel('Probability')
        # plt.xlabel('Data')
        # plt.show()
        # i += 1
        # if i >= 5:
        #     break
