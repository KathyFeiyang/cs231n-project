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
from models.submodule import stn
from test_helper import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

parser = test_re10k_parser()
args = parser.parse_args()
np.set_printoptions(precision=3)

device = "cuda" if torch.cuda.is_available() else "cpu"

flow_correct = True

def gettime():
    # get GMT time in string
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    args.savepath = args.savepath+f'/test_re10k_{gettime()}'
    log = logger.setup_logger(args.savepath + '/testing.log')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TestData, _ = D.dataloader(args.dataset, 1, args.interval, n_valid=0,
                               is_train=args.train_set, load_all_frames=True)
    TestLoader = DataLoader(DL.ImageFloder(TestData, args.dataset),
                            batch_size=1, shuffle=False, num_workers=0)

    # get auto-encoder
    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    encoder_flow = EncoderFlow(args)
    decoder_flow = Flow(args)
    flow_correction = FlowCorrection(args)
    rotate = Rotate(args)
    decoder = Decoder(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).to(device)
    encoder_traj = nn.DataParallel(encoder_traj).to(device)
    encoder_flow = nn.DataParallel(encoder_flow).to(device)
    decoder_flow = nn.DataParallel(decoder_flow).to(device)
    flow_correction = nn.DataParallel(flow_correction).to(device)
    rotate = nn.DataParallel(rotate).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))
            encoder_3d.load_state_dict(checkpoint['encoder_3d'])
            encoder_traj.load_state_dict(checkpoint['encoder_traj'])
            encoder_flow.load_state_dict(checkpoint['encoder_flow'])
            decoder_flow.load_state_dict(checkpoint['flow'])
            if flow_correct:
                flow_correction.load_state_dict(checkpoint['flow_correction'])
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
        test(TestData, TestLoader, encoder_3d, encoder_traj, encoder_flow, decoder_flow, flow_correction, decoder, rotate, log)
    log.info('full testing time = {:.2f} Minutes'.format((time.time() - start_full_time) / 60))

def test(data, dataloader, encoder_3d, encoder_traj, encoder_flow, decoder_flow, flow_correction, decoder, rotate, log):
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

        clip = video_clips[0,:frame_limit].to(device)
        t, c, h, w = clip.size()

        poses = get_poses(encoder_traj, clip)
        trajectory = construct_trajectory(poses)
        trajectory = trajectory.reshape(-1,12)

        preds = []
        for i in range(t-1):
            if i == 0:
                preds.append(clip[0:1])
                scene_rep = encoder_3d(video_clips[:, 0])
                scene_index = 0
            elif i % args.reinit_k == 0:
                # reinitialize 3d voxel
                scene_rep = encoder_3d(pred)
                # scene_rep = encoder_3d(video_clips[:, i])
                scene_index = i
            clip_in = torch.stack([clip[scene_index], clip[i+1]])
            pose = get_pose_window(encoder_traj, clip_in)
            z = euler2mat(pose[1:])
            rot_vox = stn(scene_rep, z)
            rot_codes = rotate.module.second_part(rot_vox)

            # construct flow
            final_rep = encoder_3d(video_clips[:, i+1])
            final_vox = stn(final_rep, get_pose0(encoder_traj, clip[i+1]))
            flow_rep = encoder_flow(rot_vox, final_vox)

            # flow_scaling = flow_rep.abs().max(-1)[0] / flow_rep.abs().max()
            # flow_scaling = flow_scaling.unsqueeze(1).repeat(1, 32, 1, 1, 1)
            reconstruct_voxel_partial = decoder_flow(rot_vox, flow_rep)
            if flow_correct:
                reconstruct_voxel = flow_correction(reconstruct_voxel_partial, flow_rep)
            else:
                reconstruct_voxel = reconstruct_voxel_partial
            # mark absolute difference

            sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

            flow_scaling = 1-sim(reconstruct_voxel, final_vox) #(reconstruct_voxel - final_vox).abs().mean(1)
            flow_scaling = flow_scaling / flow_scaling.max()
            flow_scaling = flow_scaling.unsqueeze(1).repeat(1, 32, 1, 1, 1)
            print(flow_scaling.mean(), flow_scaling.median())

            # flow_scaling = (reconstruct_voxel - final_vox).abs().mean(1)
            # flow_scaling = flow_scaling / flow_scaling.max()
            # flow_scaling = flow_scaling.unsqueeze(1).repeat(1, 32, 1, 1, 1)
            # print(flow_scaling.mean(), flow_scaling.median())


            reconstruct_codes = rotate.module.second_part(reconstruct_voxel) # * (flow_scaling))

            output = decoder(reconstruct_codes)
            pred = F.interpolate(output, (h, w), mode='bilinear')
            pred = torch.clamp(pred, 0, 1)
            preds.append(pred)

        # output
        synth_save_dir = os.path.join(args.savepath, f"Videos")
        os.makedirs(synth_save_dir, exist_ok=True)
        preds = torch.cat(preds,dim=0)
        pred = (preds.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(synth_save_dir+f'/video_{b_i}_pred.mp4', pred, 6)
        vid = (clip.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(synth_save_dir+f'/video_{b_i}_true.mp4', vid, 6)

        pose_save_dir = os.path.join(args.savepath, f"Poses")
        os.makedirs(pose_save_dir, exist_ok=True)
        true_camera_file = os.path.dirname(data[b_i][0]).replace('dataset_square', 'RealEstate10K')+'.txt'
        # with open(true_camera_file) as f:
        #     f.readline() # remove line 0
        #     poses = np.loadtxt(f)
        #     camera = poses[:,7:].reshape([-1,12])[:len(trajectory)]
        with open(pose_save_dir+f'/video_{b_i}_pred.txt','w') as f:
            lines = [' '.join(map(str,y))+'\n' for y in trajectory.tolist()]
            f.writelines(lines)
        # with open(pose_save_dir+f'/video_{b_i}_true.txt','w') as f:
        #     lines = [' '.join(map(str,y))+'\n' for y in camera]
        #     f.writelines(lines)
    print()

if __name__ == '__main__':
    main()
