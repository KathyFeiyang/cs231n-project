import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, xavier_normal_, kaiming_normal_
from .submodule import conv, conv3d, get_resnet50, stn

from .util import euler2mat


class Encoder3D(nn.Module):
    def __init__(self, args):
        super(Encoder3D, self).__init__()
        self.feature_extraction = get_resnet50()
        self.conv3d_1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.conv3d_2 = nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1)

    def forward(self, img):
        z_2d = self.feature_extraction(img)
        B,C,H,W = z_2d.shape
        z_3d = z_2d.reshape([-1, 256, 8, H, W])
        z_3d = F.leaky_relu(self.conv3d_1(z_3d))
        z_3d = F.leaky_relu(self.conv3d_2(z_3d))
        return z_3d

class EncoderTraj(nn.Module):
    def __init__(self, args):
        super(EncoderTraj, self).__init__()
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)

        self.scale_rotate = args.scale_rotate
        self.scale_translate = args.scale_translate

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, input):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), 6)

        pose_r = pose[:,:3] * self.scale_rotate
        pose_t = pose[:,3:] * self.scale_translate

        pose_final = torch.cat([pose_r, pose_t], 1)

        return pose_final

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

# class EncoderFlow(nn.Module):
#     def __init__(self, args):
#         super(EncoderFlow, self).__init__()
#         self.conv3d_1 = conv3d(64 * 2, 64, kernel_size=3)
#         self.conv3d_2 = conv3d(64, 32, kernel_size=3)
#         self.conv3d_3 = nn.ConvTranspose3d(32, 32, 3, stride=2)
#         self.conv3d_4 = nn.ConvTranspose3d(32, 3, 4, stride=(2, 2, 2), padding=(2, 2, 2))
#
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose3d):
#                 xavier_uniform_(m.weight.data, 5/3)
#                 if m.bias is not None:
#                     zeros_(m.bias)
#
#
#     def forward(self, transformed_start_voxel, end_voxel):
#         out = torch.cat((transformed_start_voxel, end_voxel), axis=1)
#         out = self.conv3d_1(out)
#         out = self.conv3d_2(out)
#         out = F.tanh(self.conv3d_3(out))
#         out = self.conv3d_4(out)
#         out = ContiguousGrad.apply(out)
#         out = out.permute(0, 2, 3, 4, 1)  # [1, H=32, W=64, H=64, 3]
#         return F.tanh(out)

class EncoderFlow(nn.Module):
    def __init__(self, args):
        super(EncoderFlow, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv3d_1 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3d_3 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_4 = nn.ConvTranspose3d(256, 128, 3, stride=2, bias=False)
        self.conv3d_5 = nn.ConvTranspose3d(128, 3, 4, stride=(2, 2, 2), padding=(2, 2, 2), bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                xavier_uniform_(m.weight.data, 5/3)
                if m.bias is not None:
                    zeros_(m.bias)


    def forward(self, transformed_start_voxel, end_voxel):
        out = torch.cat((transformed_start_voxel, end_voxel), axis=1)
        out = self.relu(self.conv3d_1(out))
        out = self.relu(self.conv3d_2(out))
        out = self.relu(self.conv3d_3(out))
        out = F.tanh(self.conv3d_4(out))
        out = F.tanh(self.conv3d_5(out))

        B, _, H, W, D = out.size()

        flow_resh = out.reshape(B * 3, 1, H, W, D)
        weight = torch.ones((1, 1, 3, 3, 3)) / 27
        flow_smooth = F.conv3d(flow_resh, weight, stride=1, padding=1)

        out = flow_smooth.reshape(B, 3, H, W, D)
        out = ContiguousGrad.apply(out)
        out = out.permute(0, 2, 3, 4, 1)  # [1, H=32, W=64, H=64, 3]
        return out

# class EncoderFlow(nn.Module):
#
#     def __init__(self, args):
#         super(EncoderFlow, self).__init__()
#
#         self.relu = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool3d(2, 2)
#
#         self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv1_bn = nn.BatchNorm3d(64)
#
#         self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2_bn = nn.BatchNorm3d(128)
#
#         self.bottleneck = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bottleneck_bn = nn.BatchNorm3d(128)
#
#         self.deconv1 = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.deconv1_bn = nn.BatchNorm3d(128)
#
#         self.deconv2 = nn.Conv3d(192, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.deconv2_bn = nn.BatchNorm3d(64)
#
#         self.conv3 = nn.Conv3d(64, 3, kernel_size=3, stride=1, padding=1)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight.data.normal_(0, 0.01)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, transformed_start_voxel, end_voxel):
#         x = torch.cat((transformed_start_voxel, end_voxel), axis=1)
#
#         conv1 = self.relu(self.conv1_bn(self.conv1(x)))
#         print('conv1', conv1.size())
#
#         x = self.pool(conv1)
#
#         conv2 = self.relu(self.conv2_bn(self.conv2(x)))
#         print('conv2', conv2.size())
#
#         x = self.pool(conv2)
#
#         x = self.relu(self.bottleneck_bn(self.bottleneck(x)))
#
#         print('bottleneck', x.size())
#
#         x = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
#
#         x = torch.cat((x, conv2), dim=1)
#         x = self.relu(self.deconv1_bn(self.deconv1(x)))
#         print('deconv1', x.size())
#
#         x = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
#
#         x = torch.cat((x, conv1), dim=1)
#         x = self.relu(self.deconv2_bn(self.deconv2(x)))
#         print('deconv2', x.size())
#
#         x = F.tanh(self.conv3(x))
#
#         return x.permute(0, 2, 3, 4, 1)




class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.depth_3d = 32
        self.conv3 = nn.Conv2d(2048, 512, 1)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.upconv_final = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, code):
        code = code.view(-1, code.size(1) * code.size(2), code.size(3), code.size(4))
        code = F.leaky_relu(self.conv3(code))
        code = F.leaky_relu(self.upconv3(code))
        code = F.leaky_relu(self.upconv4(code))
        output = self.upconv_final(code)
        return output

class Rotate(nn.Module):
    def __init__(self, args):
        super(Rotate, self).__init__()
        self.padding_mode = args.padding_mode
        self.conv3d_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3d_2 = nn.Conv3d(64, 64, 3, padding=1)

    def second_part(self, code):
        rot_code = F.leaky_relu(self.conv3d_1(code))
        rot_code = F.leaky_relu(self.conv3d_2(rot_code))
        return rot_code

    def forward(self, code, theta):
        rot_code = stn(code, theta, self.padding_mode)
        rot_code = F.leaky_relu(self.conv3d_1(rot_code))
        rot_code = F.leaky_relu(self.conv3d_2(rot_code))
        return rot_code

class Flow(nn.Module):
    def __init__(self, args):
        super(Flow, self).__init__()
        self.padding_mode = args.padding_mode

    def forward(self, code, grid):
        B = code.size()[0]
        r = torch.zeros(B, 6)
        theta = euler2mat(r)
        grid = grid + F.affine_grid(theta, code.size())
        rot_code = F.grid_sample(code, grid, mode='bilinear', padding_mode=self.padding_mode)

        # rot_code = rot_code.reshape(B * 32, 1, 32, 64, 64)
        # weight = torch.ones((1, 1, 3, 3, 3)) / 27
        # rot_code = F.conv3d(rot_code, weight, stride=1, padding=1)
        # rot_code = rot_code.reshape(B, 32, 32, 64, 64)
        return rot_code
