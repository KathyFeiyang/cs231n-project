import os
import time
import logger
import torch
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from parser import train_parser
import data.image_folder as D
import data.data_loader as DL
from models.autoencoder import *
from models.discriminator import *
from train_helper import *
from eval_syn_re10k import compute_error_video
from tqdm import tqdm
from test_helper import *
import numpy as np

### temp ###
n_valid = 1

parser = train_parser()
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_fraction_transform(start_frame_idx, end_frame_idx, mid_frame_idx, original):
    """Computes fraction of rotation + translation transformation."""
    fraction = (mid_frame_idx - start_frame_idx) / (end_frame_idx - start_frame_idx)
    return _get_fraction_transfor(fraction, original)


def _get_fraction_transfor(fraction, original):
    """Computes fraction of transformation."""
    return fraction * original


def gettime():
    # get GMT time in string
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())


def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    args.savepath = args.savepath+f'/flow_train_{gettime()}'
    log = logger.setup_logger(args.savepath + '/training.log')
    writer = SummaryWriter(log_dir=args.savepath)

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData, _ = D.dataloader(args.dataset, args.clip_length, args.interval, n_valid=n_valid)
    _, ValidData = D.dataloader(args.dataset, args.clip_length, args.interval, load_all_frames=True)
    log.info(f'#Train vid: {len(TrainData)}')

    print("TrainData:", len(TrainData))
    TrainLoader = DataLoader(DL.ImageFloder(TrainData, args.dataset),
        batch_size=args.bsize, shuffle=True, num_workers=args.worker, drop_last=True
    )
    ValidLoader = DataLoader(DL.ImageFloder(ValidData, args.dataset),
        batch_size=1, shuffle=False, num_workers=0,drop_last=True
    )

    # get auto-encoder
    encoder_3d = Encoder3D(args)
    encoder_traj = EncoderTraj(args)
    rotate = Rotate(args)
    encoder_flow = EncoderFlow(args)
    flow = Flow(args)
    decoder = Decoder(args)

    # get discriminator
    netd = NetD(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).to(device)
    encoder_traj = nn.DataParallel(encoder_traj).to(device)
    rotate = nn.DataParallel(rotate).to(device)
    encoder_flow = nn.DataParallel(encoder_flow).to(device)
    flow = nn.DataParallel(flow).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    all_param = list(encoder_flow.parameters()) + list(flow.parameters())

    optimizer_g = torch.optim.Adam(all_param, lr=args.lr, betas=(0,0.999))

    log.info('Number of parameters: {}'.format(sum([p.data.nelement() for p in all_param])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))
            encoder_3d.load_state_dict(checkpoint['encoder_3d'],strict=False)
            encoder_traj.load_state_dict(checkpoint['encoder_traj'],strict=False)
            # encoder_flow.load_state_doct(checkpoint['encoder_flow'], strict=False)
            # flow.load_state_doct(checkpoint['flow'], strict=False)
            decoder.load_state_dict(checkpoint['decoder'],strict=False)
            rotate.load_state_dict(checkpoint['rotate'],strict=False)
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')

    start_full_time = time.time()

    start_frame_idx = 0
    end_frame_idx = args.clip_length - 1

    for epoch in range(args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainLoader, ValidLoader, start_frame_idx, end_frame_idx,
              encoder_3d, encoder_traj, encoder_flow, decoder, flow, rotate,
              optimizer_g, log, epoch, writer)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def generate_voxels(video_clips, frame_limit, encoder_3d, encoder_traj, rotate,
         start_frame_idx, end_frame_idx):
    """Generates start voxels and end voxels.

    Assumes that the batch size of `video_clips` is 1.
    The start voxels are rotated according the trajectory from start to end."""
    full_clip = video_clips[0, :frame_limit].to(device)

    # Extract start and end frames
    end_frame_idx = min(end_frame_idx, full_clip.shape[0] - 1)
    clip = full_clip[[start_frame_idx, end_frame_idx]]
    t, c, h, w = clip.size()

    # Predict trajectory
    poses = get_poses(encoder_traj, clip)  # T x C x H x W
    trajectory = construct_trajectory(poses)  # T x 3 x 4
    trajectory = trajectory.reshape(-1, 12)  # T x 3 x 4

    start_voxel = encoder_3d(clip[0:1])
    clip_in = torch.stack([clip[0], clip[1]])  # 2 x 3 x H x W (2 frames)
    pose = get_pose_window(encoder_traj, clip_in)  # 2 x 6 (3 for r and 3 for t)
    z = euler2mat(pose[1:])  # 1 x 3 x 4 (input = 2nd frame onwards)
    start_rot_codes = rotate(start_voxel, z)

    z_identity = torch.zeros_like(z)
    for i in range(3):
        z_identity[:, i, i] = 1
    end_voxel = encoder_3d(clip[1:2])
    end_rot_codes = rotate(end_voxel, z_identity)

    return start_rot_codes, end_rot_codes


def generate_interpolation_pairs(
        video_clips, frame_limit, encoder_3d, encoder_traj, rotate):
    """Generates 2-spaced frame pairs for interpolation.
    
    Returns both the end-reference-frame rotated and middle-reference-frame
    rotated voxels for the start frame."""
    # clip = video_clips[0,:frame_limit].cuda()
    video_clips = video_clips[:, :frame_limit]

    # Extract pairs of frames with a spacing of 2
    n, t, c, h, w = video_clips.size()
    start_end_pairs = []
    start_mid_pairs = []
    for batch_idx in range(n):
        full_clip = video_clips[batch_idx: batch_idx + 1]
        for i in range(t - 3):
            subclip = full_clip[:, i: i+3]
            start_rot_codes_end_ref, end_rot_codes = generate_voxels(
                subclip[:, [0, 2]], 2, encoder_3d, encoder_traj, rotate,
                start_frame_idx=0, end_frame_idx=1)
            start_rot_codes_mid_ref, middle_rot_codes = generate_voxels(
                subclip[:, [0, 1]], 2, encoder_3d, encoder_traj, rotate,
                start_frame_idx=0, end_frame_idx=1)
            start_end_pairs.append((start_rot_codes_end_ref, end_rot_codes))
            start_mid_pairs.append((start_rot_codes_mid_ref, middle_rot_codes))

    return start_end_pairs, start_mid_pairs


cur_max_psnr = 0
def train(TrainLoader, ValidLoader, start_frame_idx, end_frame_idx,
          encoder_3d, encoder_traj, encoder_flow, decoder, flow, rotate,
          optimizer_g, log, epoch, writer):
    """Trains the model using reconstruction and interpolation loss"""
    _loss = AverageMeter()
    n_b = len(TrainLoader)
    if device == "cuda":
        torch.cuda.synchronize()
    b_s = time.perf_counter()

    encoder_3d.eval()
    encoder_traj.eval()
    rotate.eval()
    decoder.eval()

    encoder_flow.train()
    flow.train()

    video_limit = min(args.video_limit, len(TrainLoader))
    frame_limit = args.clip_length
    for b_i, video_clips in tqdm(enumerate(TrainLoader)):
        if b_i == video_limit: break

        #with autograd.detect_anomaly():
        adjust_lr(args, optimizer_g, epoch, b_i, n_b)
        video_clips = video_clips.to(device)
        n_iter = b_i + n_b * epoch

        optimizer_g.zero_grad()

        # Compute reconstruction loss
        start_rot_codes, end_rot_codes = generate_voxels(
            video_clips, frame_limit, encoder_3d, encoder_traj, rotate,
            start_frame_idx, end_frame_idx)

        reconstruction_loss = compute_reconstruction_loss_f(
            encoder_flow, flow,
            target_train=end_rot_codes, rot_codes=start_rot_codes, return_output=False)

        # Compute interpolation loss
        interpolation_loss = 0.0
        start_end_pairs, start_mid_pairs = generate_interpolation_pairs(
            video_clips, frame_limit, encoder_3d, encoder_traj, rotate)
        for start_end_pair, start_mid_pair in zip(
                start_end_pairs, start_mid_pairs):
            start_rot_codes_end_ref, end_rot_codes = start_end_pair
            start_rot_codes_mid_ref, middle_rot_codes = start_mid_pair
            interpolation_loss += compute_interpolation_loss_f(
                encoder_flow, flow, middle_rot_codes, fraction=0.5,
                start_rot_codes_end_ref=start_rot_codes_end_ref, end_rot_codes=end_rot_codes,
                start_rot_codes_mid_ref=start_rot_codes_mid_ref, return_output=False)

        # Compute total loss and back prop
        total_loss = reconstruction_loss + interpolation_loss
        total_loss.backward()
        optimizer_g.step()

        _loss.update(total_loss.item())
        batch_time = time.perf_counter() - b_s
        b_s = time.perf_counter()

        writer.add_scalar('Total Loss (Train)', total_loss, n_iter)
        info = 'Loss Image = {:.3f}({:.3f})'.format(_loss.val, _loss.avg) if _loss.count > 0 else '..'
        log.info('Epoch {} [{}/{}] {} T={:.2f}'.format(epoch, b_i, n_b, info, batch_time))

        if n_iter > 0 and n_iter % args.valid_freq == 0:
        # if True:  # for debugging purposes
            with torch.no_grad():
                val_reconstruction_loss = test_reconstruction_f(
                    ValidLoader, frame_limit, start_frame_idx, end_frame_idx,
                    encoder_3d, encoder_traj, rotate,
                    encoder_flow, flow, log, epoch, n_iter, writer)
                #output_dir = visualize_synthesis(args, ValidLoader, encoder_flow, flow,
                #                                 log, n_iter)
                #avg_psnr, _, _ = test_synthesis(output_dir)
                val_interpolation_loss = test_reconstruction_f(
                    ValidLoader, frame_limit, start_frame_idx, end_frame_idx,
                    encoder_3d, encoder_traj, rotate,
                    encoder_flow, flow, log, epoch, n_iter, writer)
            val_total_loss = val_reconstruction_loss + val_interpolation_loss
            log.info("Saving new checkpoint_flow.")
            savefilename = args.savepath + '/checkpoint_flow.tar'
            save_checkpoint(encoder_3d, encoder_traj, rotate, encoder_flow, flow, decoder, savefilename)

            global cur_min_loss
            if val_total_loss < cur_max_psnr:
                log.info("Saving new best checkpoint_flow.")
                cur_min_loss = val_total_loss
                savefilename = args.savepath + '/checkpoint_flow_best.tar'
                save_checkpoint(encoder_3d, encoder_traj, rotate, encoder_flow, flow, decoder, savefilename)


def compute_reconstruction_loss_f(encoder_flow, flow, target_train,
                                  rot_codes, return_output):
    """Computes end frame reconstruction loss."""
    flow_rep = encoder_flow(rot_codes, target_train)
    reconstruct_voxel = flow(rot_codes, flow_rep)
    loss_l1 = (reconstruct_voxel - target_train).abs().mean()
    if return_output:
        return loss_l1, reconstruct_voxel
    else:
        return loss_l1


def compute_interpolation_loss_f(encoder_flow, flow, target_middle, fraction,
                                 start_rot_codes_end_ref, end_rot_codes,
                                 start_rot_codes_mid_ref, return_output):
    """Computes middle frame interpolation loss.

    This loss enforces the flow representation to be linear and support
    meaningful scalar multiplication.
    """
    flow_rep = encoder_flow(start_rot_codes_end_ref, end_rot_codes)
    fraction_flow_rep = _get_fraction_transfor(fraction, flow_rep)
    interpolated_middle = flow(start_rot_codes_mid_ref, fraction_flow_rep)
    loss_l1 = (interpolated_middle - target_middle).abs().mean()
    if return_output:
        return loss_l1, interpolated_middle
    else:
        return loss_l1


def test_reconstruction_f(dataloader, frame_limit, start_frame_idx, end_frame_idx,
                          encoder_3d, encoder_traj, rotate,
                          encoder_flow, flow, log, epoch, n_iter, writer):
    _loss = AverageMeter()
    n_b = len(dataloader)

    encoder_flow.eval()
    flow.eval()

    for b_i, video_clips in enumerate(dataloader):
        b_s = time.perf_counter()
        video_clips = video_clips.to(device)
        n_iter = b_i + n_b * epoch

        with torch.no_grad():
            start_rot_codes, end_rot_codes = generate_voxels(
                video_clips, frame_limit, encoder_3d, encoder_traj, rotate,
                start_frame_idx, end_frame_idx)
            l_r = compute_reconstruction_loss_f(
                encoder_flow, flow,
                target_train=end_rot_codes, rot_codes=start_rot_codes, return_output=False)
        writer.add_scalar('Reconstruction Loss (Valid)', l_r, n_iter)
        _loss.update(l_r.item())
        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        log.info('Validation (recon.) at Epoch {} [{}/{}] {} T={:.2f}'.format(
            epoch, b_i, n_b, info, b_t))

    return _loss.avg


def test_interpolation_f(dataloader, frame_limit, start_frame_idx, end_frame_idx,
                         encoder_3d, encoder_traj, rotate,
                         encoder_flow, flow, log, epoch, n_iter, writer):
    _loss = AverageMeter()
    n_b = len(dataloader)

    encoder_flow.eval()
    flow.eval()

    for b_i, video_clips in enumerate(dataloader):
        b_s = time.perf_counter()
        video_clips = video_clips.to(device)
        n_iter = b_i + n_b * epoch

        with torch.no_grad():
            interpolation_loss = 0.0
            start_end_pairs, start_mid_pairs = generate_interpolation_pairs(
                video_clips, frame_limit, encoder_3d, encoder_traj, rotate,
                start_frame_idx, end_frame_idx)
            for start_end_pair, start_mid_pair in zip(
                    start_end_pairs, start_mid_pairs):
                start_rot_codes_end_ref, end_rot_codes = start_end_pair
                start_rot_codes_mid_ref, middle_rot_codes = start_mid_pair
                interpolation_loss += compute_interpolation_loss_f(
                    encoder_flow, flow, middle_rot_codes, fraction=0.5,
                    start_rot_codes_end_ref=start_rot_codes_end_ref, end_rot_codes=end_rot_codes,
                    start_rot_codes_mid_ref=start_rot_codes_mid_ref, return_output=False)
                print("interpolation loss", interpolation_loss)
        writer.add_scalar('Interpolation Loss (Valid)', l_r, n_iter)
        _loss.update(interpolation_loss.item())
        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        log.info('Validation (inter.) at Epoch {} [{}/{}] {} T={:.2f}'.format(
            epoch, b_i, n_b, info, b_t))

    return _loss.avg





# def test_reconstruction(dataloader, encoder_3d, encoder_traj, decoder, rotate, log, epoch, n_iter, writer):
#     _loss = AverageMeter()
#     n_b = len(dataloader)
#     for b_i, vid_clips in enumerate(dataloader):
#         encoder_3d.eval()
#         encoder_traj.eval()
#         decoder.eval()
#         rotate.eval()
#         b_s = time.perf_counter()
#         vid_clips = vid_clips.cuda()
#         with torch.no_grad():
#             l_r = compute_reconstruction_loss(args, encoder_3d, encoder_traj,
#                                               rotate, decoder, vid_clips)
#         writer.add_scalar('Reconstruction Loss (Valid)', l_r, n_iter)
#         _loss.update(l_r.item())
#         info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
#         b_t = time.perf_counter() - b_s
#         log.info('Validation at Epoch{} [{}/{}] {} T={:.2f}'.format(
#             epoch, b_i, n_b, info, b_t))
#     return _loss.avg
#
# def test_synthesis(output_dir):
#     values_psnr, values_ssim, values_lpips = [], [], []
#     for i in range(20):
#         video1 = output_dir+'/eval_video_{}_pred.mp4'.format(i)
#         video2 = output_dir+'/eval_video_{}_true.mp4'.format(i)
#         results = compute_error_video(video1, video2, lpips=False)
#         values_psnr.append(results['PSNR'])
#         values_ssim.append(results['SSIM'])
#         values_lpips.append(results['LPIPS'])
#
#     avg_psnr = np.mean(np.array(values_psnr))
#     avg_ssim = np.mean(np.array(values_ssim))
#     avg_lpips = np.mean(np.array(values_lpips))
#
#     return (avg_psnr, avg_ssim, avg_lpips)

if __name__ == '__main__':
    main()
