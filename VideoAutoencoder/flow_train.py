import os
import time
import logger
import torch
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
n_valid = 20

parser = train_parser()
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_fraction_transform(start_frame_idx, end_frame_idx, mid_frame_idx, original):
    """Computes fraction of rotation + translation transformation."""
    fraction = (mid_frame_idx - start_frame_idx) / (end_frame_idx - start_frame_idx)
    return original * fraction

def main():
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
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
    # TrainData_flow, target_train = prev(TrainData, TrainLoader, encoder_3d, encoder_traj, decoder, rotate, log,
    #          start_frame_idx=0, end_frame_idx=args.frame_limit - 1, mid_frame_idx=None,
    #          save_mid_only=True)
    # ValidData_flow,target_valid = prev(ValidData, ValidLoader, encoder_3d, encoder_traj, decoder, rotate, log,
    #          start_frame_idx=0, end_frame_idx=args.frame_limit - 1, mid_frame_idx=None,
    #          save_mid_only=True)

    # TrainLoader = DataLoader(DL.ImageFloder(TrainData_flow, args.dataset),
    #     batch_size=args.bsize, shuffle=True, num_workers=args.worker,drop_last=True
    # )
    # ValidLoader = DataLoader(DL.ImageFloder(ValidData_flow, args.dataset),
    #     batch_size=1, shuffle=False, num_workers=0,drop_last=True
    # )

    start_frame_idx = 0
    end_frame_idx = args.clip_length - 1

    for epoch in range(args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainLoader, ValidLoader, start_frame_idx, end_frame_idx,
              encoder_3d, encoder_traj, encoder_flow, decoder, flow, rotate,
              optimizer_g, log, epoch, writer)

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def generate_voxels(video_clips, frame_limit, encoder_3d, encoder_traj, rotate, log,
         start_frame_idx, end_frame_idx, mid_frame_idx=None, save_mid_only=True):
    ''''''
    # clip = video_clips[0,:frame_limit].cuda()
    full_clip = video_clips[0, :frame_limit]

    # Extract start and end frames
    end_frame_idx = min(end_frame_idx, full_clip.shape[0] - 1)
    if mid_frame_idx is None:
        mid_frame_idx = int(0.5 * (start_frame_idx + end_frame_idx))
    clip = full_clip[[start_frame_idx, end_frame_idx]]
    t, c, h, w = clip.size()

    # Predict trajectory
    poses = get_poses(encoder_traj, clip)  # T x C x H x W
    trajectory = construct_trajectory(poses)  # T x 3 x 4
    trajectory = trajectory.reshape(-1, 12)  # T x 3 x 4

    preds = []
    if not save_mid_only:
        # Add start frame
        preds.append(clip[0:1])

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
    print("start:", start_rot_codes.shape)
    print("end:", end_rot_codes.shape)
    return start_rot_codes, end_rot_codes

cur_max_psnr = 0
def train(TrainLoader, ValidLoader, start_frame_idx, end_frame_idx,
          encoder_3d, encoder_traj, encoder_flow, decoder, flow, rotate,
          optimizer_g, log, epoch, writer):
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
        print("\n", b_i)
        if b_i == video_limit: break

        adjust_lr(args, optimizer_g, epoch, b_i, n_b)
        video_clips = video_clips.to(device)
        n_iter = b_i + n_b * epoch

        optimizer_g.zero_grad()

        start_rot_codes, end_rot_codes = generate_voxels(
            video_clips, frame_limit, encoder_3d, encoder_traj, rotate, log,
            start_frame_idx, end_frame_idx, mid_frame_idx=None, save_mid_only=True)
        #l_r, fake_clips = compute_reconstruction_loss_f(args, encoder_3d, encoder_traj, target_train,
        #                                               rotate, decoder, video_clips,  return_output=True)
        #l_c = compute_consistency_loss(args, encoder_3d, encoder_traj, video_clips)
        #l_g = compute_gan_loss(netd, fake_clips)
        l_r = compute_reconstruction_loss_f(
            encoder_flow, flow,
            target_train=end_rot_codes, rot_codes=start_rot_codes, return_output=False)
        #sum_loss = l_r + args.lambda_voxel * l_c + args.lambda_gan * l_g
        l_r.backward()
        optimizer_g.step()

        _loss.update(l_r.item())
        batch_time = time.perf_counter() - b_s
        b_s = time.perf_counter()

        writer.add_scalar('Reconstruction Loss (Train)', l_r, n_iter)
        info = 'Loss Image = {:.3f}({:.3f})'.format(_loss.val, _loss.avg) if _loss.count > 0 else '..'
        log.info('Epoch{} [{}/{}] {} T={:.2f}'.format(epoch, b_i, n_b, info, batch_time))

        if n_iter > 0 and n_iter % args.valid_freq == 0:
            with torch.no_grad():
                val_loss = test_reconstruction_f(
                    ValidLoader, encoder_flow, flow, target_valid,
                                    log, epoch, n_iter, writer)
                #output_dir = visualize_synthesis(args, ValidLoader, encoder_flow, flow,
                #                                 log, n_iter)
                #avg_psnr, _, _ = test_synthesis(output_dir)

            log.info("Saving new checkpoint_flow.")
            savefilename = args.savepath + '/checkpoint_flow.tar'
            save_checkpoint(encoder_flow, flow, savefilename)

            global cur_min_loss
            if val_loss < cur_max_psnr:
                log.info("Saving new best checkpoint_flow.")
                cur_min_loss = val_loss
                savefilename = args.savepath + '/checkpoint_flow_best.tar'
                save_checkpoint(encoder_flow, flow, savefilename)


def compute_reconstruction_loss_f(encoder_flow, flow, target_train,
                                  rot_codes, return_output):
    flow_rep = encoder_flow(rot_codes, target_train)
    reconstruct_voxel = flow(rot_codes, flow_rep)
    loss_l1 = (reconstruct_voxel - target_train).abs().mean()
    if return_output:
        return loss_l1, reconstruct_voxel
    else:
        return loss_l1


def test_reconstruction_f(dataloader, encoder_flow, flow, target, log, epoch, n_iter, writer):
    _loss = AverageMeter()
    n_b = len(dataloader)
    for b_i, vid_clips in enumerate(dataloader):
        encoder_flow.eval()
        flow.eval()
        b_s = time.perf_counter()
        vid_clips = vid_clips.to(device)
        with torch.no_grad():
            l_r = compute_reconstruction_loss_f(encoder_flow, flow, target,
                                                vid_clips, return_output=False)
        writer.add_scalar('Reconstruction Loss (Valid)', l_r, n_iter)
        _loss.update(l_r.item())
        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        log.info('Validation at Epoch{} [{}/{}] {} T={:.2f}'.format(
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
