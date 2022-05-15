import argparse
import os
import time
import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.io as io
from parser import test_re10k_parser
import data.image_folder as D
import data.data_loader as DL
from models.autoencoder import *
from test_helper import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = test_re10k_parser()
args = parser.parse_args()
np.set_printoptions(precision=3)

device = "cuda" if torch.cuda.is_available() else "cpu"


"""Run original Video Autoencoder on middle frame interpolation.
Example usage:
`python test_re10k_baseline.py \
    --savepath log/model \
    --resume log/model/re10k.ckpt \
    --dataset HMDB51 \
    --video_limit 500
`
"""

def gettime():
    # get GMT time in string
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    args.savepath = args.savepath+f'/flow_test_{gettime()}'
    log = logger.setup_logger(args.savepath + '/testing.log')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TestData, _ = D.dataloader(args.dataset, 1, args.interval,
                               is_train=args.train_set, load_all_frames=True)
    TestLoader = DataLoader(DL.ImageFloder(TestData, args.dataset),
                            batch_size=1, shuffle=False, num_workers=0)

    # get auto-encoder
    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    encoder_flow = FlowEncoder(args)
    decoder_flow = FlowDecoder(args)
    rotate = Rotate(args)
    decoder = Decoder(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).to(device)
    encoder_traj = nn.DataParallel(encoder_traj).to(device)
    encoder_flow = nn.DataParallel(encoder_flow).to(device)
    decoder_flow = nn.DataParallel(decoder_flow).to(device)
    rotate = nn.DataParallel(rotate).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))
            encoder_3d.load_state_dict(checkpoint['encoder_3d'])
            encoder_traj.load_state_dict(checkpoint['encoder_traj'])
            encoder_flow.load_state_dict(checkpoint['encoder_flow'])
            decoder_flow.load_state_dict(checkpoint['decoder_flow'])
            decoder.load_state_dict(checkpoint['decoder'])
            rotate.load_state_dict(checkpoint['rotate'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()
    with torch.no_grad():
        log.info('start testing.')
        test(TestData, TestLoader, encoder_3d, encoder_traj, encoder_flow,
             decoder_flow, decoder, rotate, log, start_frame_idx=0,
             end_frame_idx=args.frame_limit - 1, mid_frame_idx=None,
             save_mid_only=True)
    log.info('full testing time = {:.2f} Minutes'.format((time.time() - start_full_time) / 60))

def get_fraction_transform(start_frame_idx, end_frame_idx, mid_frame_idx, original):
    """Computes fraction of rotation + translation transformation."""
    fraction = (mid_frame_idx - start_frame_idx) / (end_frame_idx - start_frame_idx)
    return original * fraction

def test(data, dataloader, encoder_3d, encoder_traj, encoder_flow, decoder_flow,
         decoder, rotate, log, start_frame_idx, end_frame_idx,
         mid_frame_idx=None, save_mid_only=True):
    _loss = AverageMeter()
    video_limit = min(args.video_limit, len(dataloader))
    frame_limit = args.frame_limit
    for b_i, video_clips in tqdm(enumerate(dataloader)):
        if b_i == video_limit: break

        encoder_3d.eval()
        encoder_traj.eval()
        encoder_flow.eval()
        decoder_flow.eval()
        decoder.eval()
        rotate.eval()

        # clip = video_clips[0,:frame_limit].cuda()
        full_clip = video_clips[0,:frame_limit]

        # Extract start and end frames
        end_frame_idx = min(end_frame_idx, full_clip.shape[0] - 1)
        if mid_frame_idx is None:
            mid_frame_idx = int(0.5 * (start_frame_idx + end_frame_idx))
        clip = full_clip[[start_frame_idx, end_frame_idx]]
        t, c, h, w = clip.size()

        # Predict trajectory
        poses = get_poses(encoder_traj, clip)  # T x C x H x W
        trajectory = construct_trajectory(poses)  # T x 3 x 4
        trajectory = trajectory.reshape(-1,12)  # T x 3 x 4

        preds = []
        if not save_mid_only:
            # Add start frame
            preds.append(clip[0:1])

        # Predict middle frame
        scene_rep = encoder_3d(video_clips[:, 0])
        scene_index = 0
        clip_in = torch.stack([clip[scene_index], clip[1]])  # 2 x 3 x H x W (2 frames)
        pose = get_pose_window(encoder_traj, clip_in)  # 2 x 6 (3 for r and 3 for t)
        pose = get_fraction_transform(
            start_frame_idx, end_frame_idx, mid_frame_idx,
            pose
        )
        z = euler2mat(pose[1:])  # 1 x 3 x 4 (input = 2nd frame onwards)
        rot_codes = rotate(scene_rep, z)

        # construct flow
        final_rep = encoder_3d(video_clips[:, end_frame_idx])
        final_codes = rotate(final_rep, get_pose0(encoder_traj, clip[1]))
        flow_rep = encoder_flow(rot_codes, final_codes)
        flow_mid = get_fraction_transform(start_frame_idx, end_frame_idx, mid_frame_idx, flow_rep)
        reconstruct_voxel = decoder_flow(rot_codes, flow_mid)

        flow_rep = encoder_flow(rot_init_voxels, final_voxels)
flow_pose = get_fraction_transform(
        start_frame_idx, end_frame_idx, mid_frame_idx,
        flow_pose)
z_flow = euler2mat(flow_rep)
recon_final_voxels = flow(rot_init_voxels, z_flow)


        output = decoder(reconstruct_voxel)
        pred = F.interpolate(output, (h, w), mode='bilinear')
        pred = torch.clamp(pred, 0, 1)
        preds.append(pred)

        if not save_mid_only:
            # Add end frame
            preds.append(clip[1:2])

        # Select ground-truth frames
        if not save_mid_only:
            gts = full_clip[[
                start_frame_idx,
                mid_frame_idx,
                end_frame_idx]]
        else:
            gts = full_clip[mid_frame_idx: mid_frame_idx + 1]

        # Save pred and ground-truth videos
        synth_save_dir = os.path.join(args.savepath, f"Videos")
        os.makedirs(synth_save_dir, exist_ok=True)
        preds = torch.cat(preds,dim=0)
        pred = (preds.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(synth_save_dir+f'/video_{b_i}_pred.mp4', pred, 6)
        vid = (gts.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(synth_save_dir+f'/video_{b_i}_true.mp4', vid, 6)

        # Save pred and gound-truth trajectories
        pose_save_dir = os.path.join(args.savepath, f"Poses")
        os.makedirs(pose_save_dir, exist_ok=True)
        # true_camera_file = os.path.dirname(data[b_i][0]).replace('dataset_square', 'RealEstate10K')+'.txt'
        # with open(true_camera_file) as f:
        #     f.readline() # remove line 0
        #     poses = np.loadtxt(f)
        #     reshaped_poses = poses[:,7:].reshape([-1,12])
        #     camera = reshaped_poses[[
        #         start_frame_idx,
        #         end_frame_idx]]
        with open(pose_save_dir+f'/video_{b_i}_pred.txt','w') as f:
            lines = [' '.join(map(str,y))+'\n' for y in trajectory.tolist()]
            f.writelines(lines)
        # with open(pose_save_dir+f'/video_{b_i}_true.txt','w') as f:
        #     lines = [' '.join(map(str,y))+'\n' for y in camera]
        #     f.writelines(lines)
    print()

if __name__ == '__main__':
    main()
