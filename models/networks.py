import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models
from .spadeResnetBlock import SPADEResnetBlock as spadeResnetBlock

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch2d':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        # norm_layer = nn.BatchNorm2d
    elif norm_type == 'batch3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
        # norm_layer = nn.BatchNorm3d
    elif norm_type == 'instance2d':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'instance3d':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class blankModule(nn.Module):
    def __init__(self, *args):
        super(blankModule, self).__init__()

    def forward(self, x):
        return x


class DimensionAvg(nn.Module):
    def __init__(self, dim=-1, keepdim=False):
        super(DimensionAvg, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        x = torch.mean(x, dim=(self.dim), keepdim=self.keepdim)
        return x


class flowwrapper(nn.Module):
    def __init__(self):
        super(flowwrapper, self).__init__()
        self.grid = None

    @torch.no_grad()
    def get_grid(self, batchsize, rows, cols, device, dtype=torch.float32):
        hor = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
        hor = hor.view(1, 1, 1, cols)
        hor = hor.expand(batchsize, 1, rows, cols)
        ver = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
        ver = ver.view(1, 1, rows, 1)
        ver = ver.expand(batchsize, 1, rows, cols)
        t_grid = torch.cat([hor, ver], 1).contiguous().detach_()
        t_grid.requires_grad = False
        return t_grid

    @torch.cuda.amp.autocast(False)
    def forward(self, x, flow):
        # flow: batch size * 2 * height * width
        N, _, H, W = x.size()
        if flow.dtype == torch.float32:
            _float32 = True
        else:
            _float32 = False
        x = x.float()
        flow = flow.float()
        if self.grid is None or self.grid.size() != flow.size():
            self.grid = self.get_grid(N, H, W, device=flow.device)
        flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                          flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], dim=1)
        final_grid = (self.grid + flow).permute(0, 2, 3, 1)
        out = F.grid_sample(x, final_grid, mode='bilinear',
                            padding_mode='border')
        return out

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(
                    1).fill_(self.real_label).to(input.dtype)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(
                    1).fill_(self.fake_label).to(input.dtype)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0).to(input.dtype)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(
                    pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.criterion = nn.KLDivLoss()

    def forward(self, x, y):
        x = F.log_softmax(x, 1)
        y = F.log_softmax(y, 1)
        loss = self.criterion(x, y)
        return loss


class VGGLoss(nn.Module):
    def __init__(self, vgg_net=None):
        super(VGGLoss, self).__init__()
        if vgg_net is None:
            self.vgg = Vgg19()
        else:
            self.vgg = vgg_net
        self.criterion = nn.L1Loss()
        self.weights = [1./32, 1./16, 1./8, 1./4, 1.]

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)
        loss = 0.
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class ResnetLoss(nn.Module):
    def __init__(self, resnet_net=None):
        super(ResnetLoss, self).__init__()
        if resnet_net is None:
            self.resnet = Resnet101()
        else:
            self.resnet = resnet_net
        self.criterion = nn.MSELoss()
        self.weights = [1., 1./2., 1./4., 1./8., 1./8.]

    def forward(self, x, y):
        x_resnet, y_resnet = self.resnet(x), self.resnet(y)
        loss = 0.
        for i in range(len(x_resnet)):
            loss += self.weights[i] * \
                self.criterion(x_resnet[i], y_resnet[i].detach())
        return loss

##############################################################################
# Generator
##############################################################################
class DisentangledExtractor(nn.Module):
    def __init__(self, opt):
        super(DisentangledExtractor, self).__init__()
        self.opt = opt
        norm_layer_2d = get_norm_layer(norm_type=opt.norm.lower()+'2d')
        norm_layer_3d = get_norm_layer(norm_type=opt.norm.lower()+'3d')

        nf = opt.ngf
        nf_prev = opt.input_nc
        seq = []
        for i in range(self.opt.n_downsample):
            seq += [
                nn.Conv2d(nf_prev, nf, kernel_size=3, stride=2, padding=1),
                norm_layer_2d(nf),
                nn.LeakyReLU(0.2, True)
            ]
            nf_prev = nf
            nf = min(nf*2, 512)
        self.image_encoder = nn.Sequential(*seq)

        # video
        nf = opt.ngf
        nf_prev = opt.input_nc
        seq = []
        for i in range(self.opt.n_downsample):
            seq += [
                nn.Conv3d(nf_prev, nf, kernel_size=3, stride=2, padding=1),
                norm_layer_3d(nf),
                nn.LeakyReLU(0.2, True)
            ]
            nf_prev = nf
            nf = min(nf*2, 512)
        seq += [DimensionAvg(2, False)]
        self.video_encoder = nn.Sequential(*seq)

        seq = [
            nn.Conv2d(nf_prev, 1024, kernel_size=5, stride=2, padding=2),
            norm_layer_2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1024, kernel_size=5, stride=2, padding=2),
            norm_layer_2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, opt.code_length,
                      kernel_size=5, stride=2, padding=2),
        ]

        self.motion_ext = nn.Sequential(*seq)

    def forward(self, x, text, input_type='image'):
        content = x
        if len(x.size()) == 4:
            content = self.image_encoder(content)
        elif len(x.size()) == 5:
            content = self.video_encoder(content)
        else:
            raise 'input_type must in [image, video]'

        motion = self.motion_ext(content).squeeze(3).squeeze(2)
        return content, motion


class SelfRecurrentPatternPredictor(nn.Module):
    def __init__(self, opt):
        super(SelfRecurrentPatternPredictor, self).__init__()
        self.opt = opt
        norm_layer_2d = get_norm_layer(norm_type=opt.norm.lower()+'2d')
        norm_layer_3d = get_norm_layer(norm_type=opt.norm.lower()+'3d')

        seq = [
            nn.Conv2d(opt.code_length, 1024,
                      kernel_size=1, stride=1, padding=0),
            # norm_layer_2d(1024),
            blankModule(1024),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            norm_layer_2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            norm_layer_2d(128),
            nn.LeakyReLU(0.2, True),
        ]

        self.motion_process = nn.Sequential(*seq)

        res_dimension = min(512, opt.ngf * (2**(self.opt.n_downsample - 1)))
        seq = [
            nn.Conv2d(res_dimension + 128, res_dimension,
                      kernel_size=3, stride=1, padding=1),
            norm_layer_2d(res_dimension),
            nn.LeakyReLU(0.2, True)
        ]
        for i in range(opt.n_block):
            seq += [Resnet_3_3_Block(res_dimension,
                                     norm_layer_2d, nn.LeakyReLU(0.2, True))]
        self.res_blk = nn.Sequential(*seq)

        seq = [
            nn.Conv2d(res_dimension, 1024, kernel_size=5, stride=2, padding=2),
            norm_layer_2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1024, kernel_size=5, stride=2, padding=2),
            norm_layer_2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, opt.code_length,
                      kernel_size=5, stride=2, padding=2),
        ]

        self.motion_ext = nn.Sequential(*seq)

        seq = [
            nn.Conv2d(res_dimension, 256, kernel_size=3, stride=1, padding=1),
            norm_layer_2d(256),
            nn.LeakyReLU(0.2, True),
        ]
        nf_prev = 256
        nf = 128
        for i in range(opt.n_downsample - 2):
            seq += [
                nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=1),
                norm_layer_2d(nf),
                nn.LeakyReLU(0.2, True),
                nn.UpsamplingBilinear2d(scale_factor=2)
            ]
            nf_prev = nf
            nf = max(64, nf // 2)
        self.element_decoder = nn.Sequential(*seq)

        seq = [
            nn.Conv3d(nf_prev//2, opt.label_class if opt.label_class >
                      0 else 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 4, 4), mode='nearest'),
            nn.Upsample(scale_factor=(opt.n_frame/2, 1, 1), mode='trilinear')
        ]
        self.seg_decoder = nn.Sequential(*seq)

        seq = [
            nn.Conv3d(nf_prev//2, 32, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(32),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=(opt.n_frame/2, 2, 2), mode='trilinear'),
            nn.Conv3d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        ]
        self.flow_decoder = nn.Sequential(*seq)

        self.motion_text_mlp = nn.Linear(128 + opt.text_nc, 128)

    def forward(self, content=None, motion=None, text=None, size=None):
        if content is None:
            assert hasattr(self, 'content')
            content = self.content
        if motion is None:
            assert hasattr(self, 'motion')
            motion = self.motion
        
        text = text.squeeze(2).squeeze(2).squeeze(2)
        motion_text = torch.cat((motion, text), 1)
        motion_text = self.motion_text_mlp(motion_text)
        motion_text = motion_text.unsqueeze(2).unsqueeze(3)

        motion = self.motion_process(motion_text)
        # motion = self.motion_process(motion.unsqueeze(2).unsqueeze(3))
        motion = F.interpolate(motion, size=content.size()[2:])
        x = torch.cat([content, motion], 1)
        x = self.res_blk(x)
        self.content = x
        self.motion = self.motion_ext(x).squeeze(3).squeeze(2)

        x = self.element_decoder(x).unsqueeze(2)  # N * C * 1 * H/4 * W/4
        # N * C/2 * 1 * H/4 * W/4, N * C/2 * 1 * H/4 * W/4
        x = torch.chunk(x, chunks=2, dim=1)
        x = torch.cat(x, 2)  # N * C/2 * 2 * H/4 * W/4

        seg = self.seg_decoder(x)  # N * nc * F * H * W
        flow = self.flow_decoder(x)  # N * 2 * F * H * W
        if size is not None:
            seg = F.interpolate(seg, size=(
                seg.size(2), *size), mode='trilinear')
            flow = F.interpolate(flow, size=(
                flow.size(2), *size), mode='trilinear')
        return seg, flow


class CoarseToFineSampler(nn.Module):
    def __init__(self, opt):
        super(CoarseToFineSampler, self).__init__()
        self.opt = opt
        norm_layer_2d = get_norm_layer(norm_type=opt.norm.lower()+'2d')
        norm_layer_3d = get_norm_layer(norm_type=opt.norm.lower()+'3d')

        self.encoder_1 = nn.Sequential(
            nn.Conv3d(opt.output_nc + (opt.label_class if opt.label_class >
                                       0 else 2), 64, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(64),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(128),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(256),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder_4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder_5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder_6 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.decoder_5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.decoder_4 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(256),
            nn.LeakyReLU(0.2, True)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(128),
            nn.LeakyReLU(0.2, True)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(64),
            nn.LeakyReLU(0.2, True)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer_3d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, opt.output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.spade6 = spadeResnetBlock(512, 512, opt)

    def forward(self, x, seg, text, size=None):
        x = torch.cat([x, seg], 1)  # N * (3 + nc) * F * H * W
        h1 = self.encoder_1(x)  # N * 64 * F * H * W
        # N * 128 * F * H/2  *  W/2
        h2 = self.encoder_2(F.avg_pool3d(h1, kernel_size=(1, 2, 2)))
        # N * 256 * F * H/4  *  W/4
        h3 = self.encoder_3(F.avg_pool3d(h2, kernel_size=(1, 2, 2)))
        # N * 512 * F * H/8  *  W/8
        h4 = self.encoder_4(F.avg_pool3d(h3, kernel_size=(1, 2, 2)))
        # N * 512 * F * H/16 *  W/16
        h5 = self.encoder_5(F.avg_pool3d(h4, kernel_size=(1, 2, 2)))

        # N * 512 * F * H/32 *  W/32
        h6 = self.encoder_6(F.avg_pool3d(h5, kernel_size=(1, 2, 2)))
        h6 = F.interpolate(h6, scale_factor=(1, 2, 2))
        h6 = self.spade6(h6, text)

        # N * 512 * F * H/16 *  W/16
        u5 = self.decoder_5(
            h6 + F.interpolate(h5, size=h6.size()[2:], mode='trilinear'))
        u5 = F.interpolate(u5, scale_factor=(1, 2, 2))

        # N * 256 * F * H/8  *  W/8
        u4 = self.decoder_4(
            u5 + F.interpolate(h4, size=u5.size()[2:], mode='trilinear'))
        u4 = F.interpolate(u4, scale_factor=(1, 2, 2))

        # N * 128 * F * H/4  *  W/4
        u3 = self.decoder_3(
            u4 + F.interpolate(h3, size=u4.size()[2:], mode='trilinear'))
        u3 = F.interpolate(u3, scale_factor=(1, 2, 2))

        # N * 64  * F * H/2  *  W/2
        u2 = self.decoder_2(
            u3 + F.interpolate(h2, size=u3.size()[2:], mode='trilinear'))
        u2 = F.interpolate(u2, scale_factor=(1, 2, 2))

        # N * 3   * F * H    *  W
        u1 = self.decoder_1(
            u2 + F.interpolate(h1, size=u2.size()[2:], mode='trilinear'))
        if size is not None:
            u1 = F.interpolate(u1, size=(u1.size(2), *size), mode='trilinear')
        # print(torch.max(u2), torch.min(u2))
        return u1

##############################################################################
# Discriminator
##############################################################################
class LocalDiscriminator(nn.Module):
    def __init__(self, opt):
        super(LocalDiscriminator, self).__init__()
        self. opt = opt
        norm_layer_2d = get_norm_layer(norm_type=opt.norm.lower()+'2d')
        norm_layer_3d = get_norm_layer(norm_type=opt.norm.lower()+'3d')

        self.module = nn.ModuleList()

        nf_prev = opt.output_nc
        nf = opt.ndf
        for i in range(opt.D_n_layers):
            seq = [
                nn.Conv3d(nf_prev, nf, kernel_size=(1, 3, 3),
                          stride=(1, 2, 2), padding=(0, 1, 1)),
                norm_layer_3d(nf),
                nn.LeakyReLU(0.2, True),
            ]
            nf_prev = nf
            nf = min(nf_prev * 2, 512)
            self.module.append(nn.Sequential(*seq))

        seq = [
            nn.Conv3d(nf_prev, 1, kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
        ]
        self.module.append(nn.Sequential(*seq))

    def forward(self, x):
        predict = [x]
        for m in self.module:
            predict.append(m(predict[-1]))
        return predict[1:]


class GlobalDiscriminator(nn.Module):
    def __init__(self, opt):
        super(GlobalDiscriminator, self).__init__()
        self. opt = opt
        norm_layer_2d = get_norm_layer(norm_type=opt.norm.lower()+'2d')
        norm_layer_3d = get_norm_layer(norm_type=opt.norm.lower()+'3d')

        self.module = nn.ModuleList()

        nf_prev = opt.output_nc
        nf = opt.ndf
        for i in range(opt.D_n_layers):
            seq = [
                nn.Conv3d(nf_prev, nf, kernel_size=(3, 3, 3),
                          stride=(2, 2, 2), padding=(1, 1, 1)),
                norm_layer_3d(nf),
                nn.LeakyReLU(0.2, True),
            ]
            nf_prev = nf
            nf = min(nf_prev * 2, 512)
            self.module.append(nn.Sequential(*seq))

        seq = [
            nn.Conv3d(nf_prev, 1, kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
        ]
        self.module.append(nn.Sequential(*seq))

    def forward(self, x):
        predict = [x]
        for m in self.module:
            predict.append(m(predict[-1]))

        return predict[1:]

##############################################################################
# Block & Module
##############################################################################
class Resnet_3_3_Block(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(Resnet_3_3_Block, self).__init__()
        model = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            model += [
                nn.Dropout(0.5),
            ]
        model += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            norm_layer(dim),
            activation,
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out


class Resnet_1_3_1_Block(nn.Module):
    def __init__(self, dim, norm_layer, hidden_dim=None, activation=nn.ReLU(True), use_dropout=False):
        super(Resnet_1_3_1_Block, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
        model = [
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            norm_layer(hidden_dim),
            activation,
        ]
        if use_dropout:
            model += [
                nn.Dropout(0.5),
            ]
        model += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            norm_layer(hidden_dim),
            activation,
        ]
        if use_dropout:
            model += [
                nn.Dropout(0.5),
            ]
        model += [
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            norm_layer(dim),
            activation,
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out

##############################################################################
# Perceptual Network
##############################################################################
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # for x in range(4):
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 18):
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(18, 27):
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(27, 36):
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Resnet101(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet101, self).__init__()
        resnet_pretrained = models.resnet101(pretrained=True)
        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice0.add_module('conv1', resnet_pretrained.conv1)
        self.slice0.add_module('bn1', resnet_pretrained.bn1)
        self.slice0.add_module('relu', resnet_pretrained.relu)
        self.slice0.add_module('maxpool', resnet_pretrained.maxpool)
        self.slice1.add_module('layer1', resnet_pretrained.layer1)
        self.slice2.add_module('layer2', resnet_pretrained.layer2)
        self.slice3.add_module('layer3', resnet_pretrained.layer3)
        self.slice4.add_module('layer4', resnet_pretrained.layer4)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu0 = self.slice0(X)
        h_relu1 = self.slice1(h_relu0)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        out = [h_relu0, h_relu1, h_relu2, h_relu3, h_relu4]
        return out
