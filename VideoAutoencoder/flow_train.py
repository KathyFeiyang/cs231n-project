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
from models.submodule import stn
from train_helper import *
from eval_syn_re10k import compute_error_video
from tqdm import tqdm
from test_helper import *
import numpy as np
import warnings

### temp ###
n_valid = 10
debug = True
interpolation_only = False
reconstruction_only = False
recon_train_epochs = 5
inter_train_epochs = 3
seq_length = 4
batch_size = 1
assert reconstruction_only or seq_length >= 3

parser = train_parser()
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


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

    if debug:
        args.bsize = batch_size
        print(f"TrainData len: {len(TrainData)}")
        print(f"ValidData len: {len(ValidData)}")
        # TrainData = TrainData[:1]  # for debugging purposes
        # ValidData = ValidData[:1]  # for debugging purposes
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
    flow_correction = FlowCorrection(args)
    decoder = Decoder(args)

    # get discriminator
    netd = NetD(args)

    # cuda
    encoder_3d = nn.DataParallel(encoder_3d).to(device)
    encoder_traj = nn.DataParallel(encoder_traj).to(device)
    rotate = nn.DataParallel(rotate).to(device)
    encoder_flow = nn.DataParallel(encoder_flow).to(device)
    flow = nn.DataParallel(flow).to(device)
    flow_correction = nn.DataParallel(flow_correction).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    all_param = list(encoder_flow.parameters()) + list(flow.parameters()) + list(flow_correction.parameters())

    optimizer_g = torch.optim.Adam(all_param, lr=args.lr, betas=(0.9, 0.999))

    log.info('Number of parameters: {}'.format(sum([p.data.nelement() for p in all_param])))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))
            encoder_3d.load_state_dict(checkpoint['encoder_3d'],strict=False)
            encoder_traj.load_state_dict(checkpoint['encoder_traj'],strict=False)
            # encoder_flow.load_state_dict(checkpoint['encoder_flow'], strict=False)
            # flow.load_state_dict(checkpoint['flow'], strict=False)
            # flow_correction.load_state_dict(checkpoint['flow_correction'], strict=False)
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
              encoder_3d, encoder_traj, encoder_flow, decoder, flow, flow_correction, rotate,
              optimizer_g, log, epoch, writer)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def compute_reconstruction_loss_flow(args, encoder_3d, encoder_traj, rotate,
                                     encoder_flow, flow, flow_correction,
                                     decoder, clips, optimizer, epochs=5):
    if debug:
        print("\nComputing reconstruction loss...")

    b, t, c, h, w = clips.size()
    rot_codes = None
    gt_codes = None
    # want no gradient
    with torch.no_grad():
        codes = encoder_3d(clips.reshape(b * t, c, h, w))
        _, C, H, W, D = codes.size()
        codes = codes.reshape(b, t, C, H, W, D)
        if debug:
            pass
            # print(codes.size())
            # print("- Generated all codes")

        clipsi = clips.unsqueeze(1).repeat(1, t, 1, 1, 1, 1)
        clipsj = clips.unsqueeze(2).repeat(1, 1, t, 1, 1, 1)
        clipsi = clipsi.reshape(b, t * t, c, h, w)
        clipsj = clipsj.reshape(b, t * t, c, h, w)
        if debug:
            pass
            # print("- Generated all clips")

        clips_pair = torch.cat([clipsi, clipsj], dim=2) # b x t*t x 2*c x h x w
        pair_tensor = clips_pair.reshape(b * (t * t), c*2, h, w)
        poses = encoder_traj(pair_tensor)
        theta = euler2mat(poses) # (b * t * t, 3, 4)
        if debug:
            pass
            # print("- Generated all theta")

        code_t = codes.unsqueeze(1).repeat(1, t, 1, 1, 1, 1, 1).reshape(b * t * t, C, H, W, D)
        rot_codes = stn(code_t, theta, args.padding_mode)
        _, C1, H1, W1, D1 = rot_codes.size()
        reshape_rot_codes = rot_codes.reshape(b, t, t, C1, H1, W1, D1)
        if debug:
            pass
            # print("- Did all rotations")

        gt_codes = reshape_rot_codes.diagonal(dim1=1, dim2=2).permute(0, 5, 1, 2, 3, 4)
        gt_codes = gt_codes.unsqueeze(2).repeat(1, 1, t, 1, 1, 1, 1)
        gt_codes = gt_codes.reshape(b * t * t, C1, H1, W1, D1)
        if debug:
            pass
            # print("- Generated all gt codes")

    if optimizer is None:
        epochs = 1
        gt_mask = None
    else:  # Avoids recomputing the gt_mask for epochs number of times
        gt_mask = get_mask(gt_codes, rotate, decoder)

    for i in range(epochs):
        if debug and optimizer:
            print(f"Training substep {i + 1}/{epochs}")
        if optimizer is not None:
            optimizer.zero_grad()

        loss = compute_losses(
            rot_codes, gt_codes, "reconstruction",
            encoder_flow, rotate, flow, flow_correction, decoder,
            gt_mask=gt_mask, compute_pred_mask=(i == 0 and optimizer is not None))

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return loss


def compute_losses(rot_codes, gt_codes, loss_name,
                   encoder_flow, rotate, flow, flow_correction, decoder,
                   gt_mask=None, compute_pred_mask=True):
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    gt_codes = gt_codes.to(device)
    rot_codes = rot_codes.to(device)
    flow_rep = encoder_flow(rot_codes, gt_codes)
    reconstructed_codes_partial = flow(rot_codes, flow_rep)
    reconstructed_codes = flow_correction(reconstructed_codes_partial, flow_rep)
    if compute_pred_mask:
        reconstruct_mask_partial = get_mask(reconstructed_codes_partial, rotate, decoder)
        reconstruct_mask = get_mask(reconstructed_codes, rotate, decoder)
        if gt_mask is None:
            gt_mask = get_mask(gt_codes, rotate, decoder)
        mask_partial = torch.max(reconstruct_mask_partial, gt_mask)
        mask = torch.max(reconstruct_mask, gt_mask)
    else:
        mask_partial = torch.ones_like(reconstructed_codes)
        mask = torch.ones_like(reconstructed_codes)
    reg = flow_rep.abs().mean()

    reconstructed_codes_partial1 = clip_vox(reconstructed_codes_partial) * mask_partial
    gt_codes1 = clip_vox(gt_codes) * mask_partial
    rot_codes1 = clip_vox(rot_codes) * mask_partial

    reconstructed_codes1 = clip_vox(reconstructed_codes) * mask
    gt_codes2 = clip_vox(gt_codes) * mask
    rot_codes2 = clip_vox(rot_codes) * mask

    l1_loss_partial = (reconstructed_codes_partial1 - gt_codes1).abs().mean()
    cos_sim_partial = (1 - sim(reconstructed_codes_partial1, gt_codes1)).mean()
    partial_loss = l1_loss_partial * 1e3 + cos_sim_partial

    l1_loss = (reconstructed_codes1 - gt_codes2).abs().mean()
    cos_sim = (1 - sim(reconstructed_codes1, gt_codes2)).mean()
    full_loss = l1_loss * 1e3 + cos_sim
    print(f"Uncorrected {loss_name} loss")
    l1_default_loss = (gt_codes1 - rot_codes1).abs().mean()
    l2_loss_partial = (reconstructed_codes_partial1 - gt_codes1).square().mean()
    l2_default_loss = (rot_codes1 - gt_codes1).square().mean()
    print(' - loss l2:', l2_loss_partial.item(), '\n',
            '- default l2:', l2_default_loss.item(), '\n',
            '- loss l2 diff (want positive):', ((l2_default_loss - l2_loss_partial) / l2_default_loss).item(), '\n',
            '- reg:', reg.item(), '\n',
            '- loss l1:', l1_loss_partial.item(), '\n',
            '- default l1:', l1_default_loss.item(), '\n',
            '- loss l1 diff (want positive):', ((l1_default_loss - l1_loss_partial) / l1_default_loss).item(), '\n',
            '- cos sim:', cos_sim_partial.item())

    print(f"Full {loss_name} loss")
    l1_default_loss = (gt_codes2 - rot_codes2).abs().mean()
    l2_loss = (reconstructed_codes1 - gt_codes2).square().mean()
    l2_default_loss = (rot_codes2 - gt_codes2).square().mean()
    print(' - loss l2:', l2_loss.item(), '\n',
            '- default l2:', l2_default_loss.item(), '\n',
            '- loss l2 diff (want positive):', ((l2_default_loss - l2_loss) / l2_default_loss).item(), '\n',
            '- reg:', reg.item(), '\n',
            '- loss l1:', l1_loss.item(), '\n',
            '- default l1:', l1_default_loss.item(), '\n',
            '- loss l1 diff (want positive):', ((l1_default_loss - l1_loss) / l1_default_loss).item(), '\n',
            '- cos sim:', cos_sim.item())

    loss = full_loss + partial_loss
    return loss


def compute_perceptual_loss(codes, rotate, decoder, target_imgs):
    _, c, h, w = target_imgs.size()
    output = decoder(rotate.module.second_part(codes))
    output = F.interpolate(output, (h, w), mode='bilinear')
    return perceptual_loss(output, target_imgs)


def get_mask(vox, rotate, decoder):
    vox = vox.detach()
    vox.requires_grad = True

    rot_codes = rotate.module.second_part(vox)
    reconstruct_im = decoder(rot_codes)
    full_reconstruct_im = torch.clamp(reconstruct_im, 0, 1)

    importance_loss = full_reconstruct_im.mean()

    importance_loss.backward()

    return vox.grad.abs() / vox.grad.abs().max()


def clip_vox(vox):
    histo, _ = torch.max(vox.abs(), 1, keepdim=True)
    histo_rep = histo.repeat(1, 32, 1, 1, 1)
    histo_scale = (histo_rep > 0.005)
    return vox * histo_scale


def train_on_interpolation_loss(video_clips, encoder_3d, encoder_traj,
                                encoder_flow, rotate, flow, flow_correction,
                                decoder, optimizer, epochs=3):
    total_loss = torch.tensor([0.0], requires_grad=True if optimizer else False).to(device)

    for clips in [video_clips, torch.flip(video_clips, dims=(1,))]:
        # Train and original and reversed video

        with torch.no_grad():
            start_rot_codes_end_ref, end_rot_codes, \
                start_rot_codes_mid_ref, middle_rot_codes = \
                    generate_interpolation_pairs_batch(clips, encoder_3d, encoder_traj)

        if optimizer is None:
            epochs = 1
            gt_mask = None
        else:  # Avoids recomputing the gt_mask for epochs number of times
            gt_mask = get_mask(middle_rot_codes, rotate, decoder)

        for i in range(epochs):
            if debug and optimizer:
                print(f"Training substep {i + 1}/{epochs}")
            if optimizer is not None:
                optimizer.zero_grad()

            loss = compute_interpolation_loss_f(
                encoder_flow, rotate, flow, flow_correction, decoder,
                middle_rot_codes, fraction=0.5, is_train=(i == 0 and optimizer is not None),
                gt_mask=gt_mask,
                start_rot_codes_end_ref=start_rot_codes_end_ref, end_rot_codes=end_rot_codes,
                start_rot_codes_mid_ref=start_rot_codes_mid_ref, return_output=False)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        total_loss = total_loss + loss

    return total_loss


def generate_interpolation_pairs_batch(clips, encoder_3d, encoder_traj):
    """Generates 2-spaced frame pairs for interpolation.

    Returns both the end-reference-frame rotated and middle-reference-frame
    rotated voxels for the start frame in shape b * (t-2) x c x h x w x d."""
    if debug:
        print("\nComputing interpolation loss...")
    b, t, c, h, w = clips.size()
    start_rot_codes_end_ref = None
    end_rot_codes = None
    start_rot_codes_mid_ref = None
    middle_rot_codes = None

    codes = encoder_3d(clips.reshape(b * t, c, h, w))
    _, C, H, W, D = codes.size()
    codes = codes.view(b, t, C, H, W, D)
    if debug:
        pass
        # print(codes.size())
        # print("- Generated all codes")

    clips_start = clips[:, :t-2]
    clips_end = clips[:, 2:]
    clips_middle = clips[:, 1: t-1]
    if debug:
        pass
        # print("- Generated all start, end, middle clips")

    start_end_pair = torch.cat([clips_start, clips_end], dim=2)  # b x t-2 x 2*c x h x w
    start_mid_pair = torch.cat([clips_start, clips_middle], dim=2)
    start_end_pair_tensor = start_end_pair.view(b * (t - 2), c * 2, h, w)
    start_mid_pair_tensor = start_mid_pair.view(b * (t - 2), c * 2, h, w)
    start_end_poses = encoder_traj(start_end_pair_tensor)
    start_mid_poses = encoder_traj(start_mid_pair_tensor)
    start_end_theta = euler2mat(start_end_poses)  # b * (t-2) x 3 x 4
    start_mid_theta = euler2mat(start_mid_poses)  # b * (t-2) x 3 x 4
    if debug:
        pass
        # print("- Generated all start-middle, start-end theta")

    code_start = codes[:, :t-2].reshape(b * (t - 2), C, H, W, D)
    start_rot_codes_end_ref = stn(code_start, start_end_theta, args.padding_mode)
    start_rot_codes_mid_ref = stn(code_start, start_mid_theta, args.padding_mode)
    if debug:
        pass
        # print("- Did end referenced and middle referenced rotations")

    code_end = codes[:, 2:].reshape(b * (t - 2), C, H, W, D)
    code_middle = codes[:, 1: t-1].reshape(b * (t - 2), C, H, W, D)
    theta_identity = torch.zeros_like(start_end_theta)
    for i in range(3):
        theta_identity[:, i, i] = 1
    end_rot_codes = stn(code_end, theta_identity, args.padding_mode)
    middle_rot_codes = stn(code_middle, theta_identity, args.padding_mode)
    if debug:
        pass
        # print("- Generated end and middle codes")

    return start_rot_codes_end_ref, end_rot_codes, \
            start_rot_codes_mid_ref, middle_rot_codes


def compute_interpolation_loss_f(encoder_flow, rotate, flow, flow_correction, decoder,
                                 target_middle, fraction, is_train, gt_mask,
                                 start_rot_codes_end_ref, end_rot_codes,
                                 start_rot_codes_mid_ref, return_output):
    """Computes middle frame interpolation loss.

    This loss enforces the flow representation to be linear and support
    meaningful scalar multiplication.
    """
    flow_rep = encoder_flow(start_rot_codes_end_ref, end_rot_codes)
    fraction_flow_rep = _get_fraction_transfor(fraction, flow_rep)
    interpolated_middle_partial = flow(start_rot_codes_mid_ref, fraction_flow_rep)
    interpolated_middle = flow_correction(interpolated_middle_partial, fraction_flow_rep)
    # loss_l1 = (interpolated_middle - target_middle).abs().mean()
    # rotate_only_loss_l1 = (start_rot_codes_mid_ref - target_middle).abs().mean()
    loss = compute_losses(
        interpolated_middle, target_middle, "interpolation",
        encoder_flow, rotate, flow, flow_correction, decoder,
        gt_mask=gt_mask, compute_pred_mask=is_train)
    if return_output:
        return loss, interpolated_middle
    else:
        return loss


cur_max_psnr = 0
def train(TrainLoader, ValidLoader, start_frame_idx, end_frame_idx,
          encoder_3d, encoder_traj, encoder_flow, decoder, flow, flow_correction,
          rotate, optimizer_g, log, epoch, writer):
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
    flow_correction.train()

    video_limit = min(args.video_limit, len(TrainLoader))
    frame_limit = args.clip_length
    for b_i, video_clips in tqdm(enumerate(TrainLoader)):
        if b_i == video_limit: break

        #with autograd.detect_anomaly():
        adjust_lr(args, optimizer_g, epoch, b_i, n_b)
        video_clips = video_clips.to(device)
        n_iter = b_i + n_b * epoch

        if debug:
            video_clips = video_clips[0:batch_size, 0:seq_length]  # for debugging purposes

        # Compute reconstruction loss and back prop
        reconstruction_loss = torch.tensor([0.0], requires_grad=True).to(device)
        if not interpolation_only:
            reconstruction_loss = compute_reconstruction_loss_flow(
                args, encoder_3d, encoder_traj, rotate, encoder_flow, flow, flow_correction, decoder,
                video_clips, optimizer_g, epochs=recon_train_epochs)
            log.info('Epoch {} [{}/{}] Rec Loss = {}'.format(
                epoch, b_i, n_b, reconstruction_loss))

        # Compute interpolation loss and back prop
        interpolation_loss = torch.tensor([0.0], requires_grad=True).to(device)
        if not reconstruction_only:
            interpolation_loss = train_on_interpolation_loss(
                video_clips, encoder_3d, encoder_traj, encoder_flow, rotate,
                flow, flow_correction, decoder, optimizer_g, epochs=inter_train_epochs)
            log.info('Epoch {} [{}/{}] Int Loss = {}'.format(
                epoch, b_i, n_b, interpolation_loss))

        # Compute total loss and back prop
        total_loss = reconstruction_loss + interpolation_loss

        _loss.update(total_loss.item())
        batch_time = time.perf_counter() - b_s
        b_s = time.perf_counter()

        writer.add_scalar('Total Loss (Train)', total_loss, n_iter)
        info = 'Loss Image = {:.6f}({:.6f})'.format(_loss.val, _loss.avg) if _loss.count > 0 else '..'
        log.info('Epoch {} [{}/{}] {} T={:.2f}'.format(epoch, b_i, n_b, info, batch_time))

        if (n_iter > 0 and n_iter % args.valid_freq == 0):  # for debugging purposes
            with torch.no_grad():
                # val_reconstruction_loss = test_reconstruction_f(
                #     ValidLoader, frame_limit, start_frame_idx, end_frame_idx,
                #     encoder_3d, encoder_traj, rotate,
                #     encoder_flow, flow, log, epoch, n_iter, writer)
                #output_dir = visualize_synthesis(args, ValidLoader, encoder_flow, flow,
                #                                 log, n_iter)
                #avg_psnr, _, _ = test_synthesis(output_dir)
                val_total_loss = test_f(
                    ValidLoader, frame_limit, start_frame_idx, end_frame_idx,
                    encoder_3d, encoder_traj, rotate,
                    encoder_flow, flow, flow_correction, decoder, log, epoch, n_iter, writer)
            log.info("Saving new checkpoint_flow.")
            savefilename = args.savepath + '/checkpoint_flow.tar'
            save_checkpoint(encoder_3d, encoder_traj, rotate, encoder_flow, flow, flow_correction, decoder, savefilename)

            global cur_min_loss
            if val_total_loss < cur_max_psnr:
                log.info("Saving new best checkpoint_flow.")
                cur_min_loss = val_total_loss
                savefilename = args.savepath + '/checkpoint_flow_best.tar'
                save_checkpoint(encoder_3d, encoder_traj, rotate, encoder_flow, flow, flow_correction, decoder, savefilename)
        log.info("-" * 30)


def test_f(dataloader, frame_limit, start_frame_idx, end_frame_idx,
            encoder_3d, encoder_traj, rotate,
            encoder_flow, flow, flow_correction, decoder, log, epoch, n_iter, writer):
    print("\nRunning validation...")
    _loss = AverageMeter()
    n_b = len(dataloader)

    encoder_flow.eval()
    flow.eval()
    flow_correction.eval()

    for b_i, video_clips in enumerate(dataloader):
        b_s = time.perf_counter()
        video_clips = video_clips.to(device)
        n_iter = b_i + n_b * epoch

        if debug:
            video_clips = video_clips[0:batch_size, 0:seq_length]  # for debugging purposes
        with torch.no_grad():
            reconstruction_loss = torch.tensor([0.0], requires_grad=False).to(device)
            if not interpolation_only:
                reconstruction_loss = compute_reconstruction_loss_flow(
                    args, encoder_3d, encoder_traj, rotate, encoder_flow, flow, flow_correction, decoder,
                    video_clips, optimizer=None)

            interpolation_loss = torch.tensor([0.0], requires_grad=False).to(device)
            if not reconstruction_only:
                interpolation_loss = train_on_interpolation_loss(
                    video_clips, encoder_3d, encoder_traj, encoder_flow, rotate,
                    flow, flow_correction, decoder, optimizer=None)

            total_loss = reconstruction_loss + interpolation_loss
        writer.add_scalar('Total Loss (Valid)', total_loss, n_iter)
        _loss.update(total_loss.item())
        info = 'Loss = {:.6f}({:.6f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        log.info('Validation (total) at Epoch {} [{}/{}] {} T={:.2f}'.format(
            epoch, b_i, n_b, info, b_t))

    return _loss.avg


### scratch work

# def generate_voxels(video_clips, frame_limit, encoder_3d, encoder_traj, rotate,
#          start_frame_idx, end_frame_idx):
#     """Generates start voxels and end voxels.

#     Assumes that the batch size of `video_clips` is 1.
#     The start voxels are rotated according the trajectory from start to end."""
#     full_clip = video_clips[0, :frame_limit].to(device)

#     # Extract start and end frames
#     end_frame_idx = min(end_frame_idx, full_clip.shape[0] - 1)
#     clip = full_clip[[start_frame_idx, end_frame_idx]]
#     t, c, h, w = clip.size()

#     # Predict trajectory
#     poses = get_poses(encoder_traj, clip)  # T x C x H x W
#     trajectory = construct_trajectory(poses)  # T x 3 x 4
#     trajectory = trajectory.reshape(-1, 12)  # T x 3 x 4

#     start_voxel = encoder_3d(clip[0:1])
#     clip_in = torch.stack([clip[0], clip[1]])  # 2 x 3 x H x W (2 frames)
#     pose = get_pose_window(encoder_traj, clip_in)  # 2 x 6 (3 for r and 3 for t)
#     z = euler2mat(pose[1:])  # 1 x 3 x 4 (input = 2nd frame onwards)
#     start_rot_codes = rotate(start_voxel, z)

#     z_identity = torch.zeros_like(z)
#     for i in range(3):
#         z_identity[:, i, i] = 1
#     end_voxel = encoder_3d(clip[1:2])
#     end_rot_codes = rotate(end_voxel, z_identity)

#     return start_rot_codes, end_rot_codes


# def generate_interpolation_pairs(
#         video_clips, frame_limit, encoder_3d, encoder_traj, rotate):
#     """Generates 2-spaced frame pairs for interpolation.

#     Returns both the end-reference-frame rotated and middle-reference-frame
#     rotated voxels for the start frame."""
#     # clip = video_clips[0,:frame_limit].cuda()
#     video_clips = video_clips[:, :frame_limit]

#     # Extract pairs of frames with a spacing of 2
#     n, t, c, h, w = video_clips.size()
#     start_end_pairs = []
#     start_mid_pairs = []
#     for batch_idx in range(n):
#         full_clip = video_clips[batch_idx: batch_idx + 1]
#         for i in range(t - 3):
#             subclip = full_clip[:, i: i+3]
#             start_rot_codes_end_ref, end_rot_codes = generate_voxels(
#                 subclip[:, [0, 2]], 2, encoder_3d, encoder_traj, rotate,
#                 start_frame_idx=0, end_frame_idx=1)
#             start_rot_codes_mid_ref, middle_rot_codes = generate_voxels(
#                 subclip[:, [0, 1]], 2, encoder_3d, encoder_traj, rotate,
#                 start_frame_idx=0, end_frame_idx=1)
#             start_end_pairs.append((start_rot_codes_end_ref, end_rot_codes))
#             start_mid_pairs.append((start_rot_codes_mid_ref, middle_rot_codes))

#     return start_end_pairs, start_mid_pairs

# def compute_reconstruction_loss_f(encoder_flow, flow, target_train,
#                                   rot_codes, return_output):
#     """Computes end frame reconstruction loss."""
#     flow_rep = encoder_flow(rot_codes, target_train)
#     reconstruct_voxel = flow(rot_codes, flow_rep)
#     loss_l1 = (reconstruct_voxel - target_train).abs().mean()
#     if return_output:
#         return loss_l1, reconstruct_voxel
#     else:
#         return loss_l1

# def test_reconstruction_f(dataloader, frame_limit, start_frame_idx, end_frame_idx,
#                           encoder_3d, encoder_traj, rotate, decoder,
#                           encoder_flow, flow, log, epoch, n_iter, writer):
#     _loss = AverageMeter()
#     n_b = len(dataloader)

#     encoder_3d.eval()
#     encoder_traj.eval()
#     encoder_flow.eval()
#     flow.eval()
#     rotate.eval()
#     decoder.eval()

#     for b_i, video_clips in enumerate(dataloader):
#         b_s = time.perf_counter()
#         video_clips = video_clips.to(device)
#         n_iter = b_i + n_b * epoch

#         with torch.no_grad():
#             reconstruction_loss = compute_reconstruction_loss_flow(
#                 args, encoder_3d, encoder_traj, rotate, decoder, encoder_flow, flow,
#                 video_clips, optimizer_g)
#         writer.add_scalar('Reconstruction Loss (Valid)', l_r, n_iter)
#         _loss.update(l_r.item())
#         info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
#         b_t = time.perf_counter() - b_s
#         log.info('Validation (recon.) at Epoch {} [{}/{}] {} T={:.2f}'.format(
#             epoch, b_i, n_b, info, b_t))

#     return _loss.avg

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
