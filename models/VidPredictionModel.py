import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks as networks
import util.util as util
from transformers import T5Tokenizer, T5Model

class VidPredictionModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor


        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.netT5 = T5Model.from_pretrained("t5-small")
        self.text_projection = nn.Parameter(torch.rand(512, 512))
        
        self.netExtractor, self.netPredictor, self.netSampler, \
            self.netD_local, self.netD_global = self.initialize_networks(opt)

        self.resample = networks.flowwrapper()

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not self.opt.no_feature_loss:
                if not self.opt.use_resnet:
                    self.criterionFeature = networks.VGGLoss()
                else:
                    self.criterionFeature = networks.ResnetLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.

    def forward(self, data: dict, mode: str):
        __data = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                __data['input_images'], __data['gen_images'], __data['gen_segs'], __data['text'])
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                __data['input_images'], __data['gen_images'])
            return d_loss
        elif mode == 'inference':
            fake_images = self.inference(__data['input_images'], __data['text'])
            return fake_images
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netExtractor.parameters(
        )) + list(self.netPredictor.parameters()) + list(self.netSampler.parameters())
        D_params = list(self.netD_local.parameters()) + \
            list(self.netD_global.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = opt.lr, opt.lr

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch: int):
        # util.save_network(self.netExtractor, 'Extractor', epoch, self.opt)
        # util.save_network(self.netPredictor, 'Predictor', epoch, self.opt)
        # util.save_network(self.netSampler, 'Sampler', epoch, self.opt)
        # util.save_network(self.text_projection, 'projection', epoch, self.opt)
        # util.save_network(self.netD_local, 'D_local', epoch, self.opt)
        # util.save_network(self.netD_global, 'D_global', epoch, self.opt)
        util.save_network(self, 'VidPredictionModel', epoch, self.opt)
        

    ############################################################################
    # Private helper methods
    ############################################################################
    def initialize_networks(self, opt):
        netExtractor = networks.DisentangledExtractor(opt)
        netPredictor = networks.SelfRecurrentPatternPredictor(opt)
        netSampler = networks.CoarseToFineSampler(opt)
        netExtractor.apply(networks.weights_init)
        netPredictor.apply(networks.weights_init)
        netSampler.apply(networks.weights_init)

        if opt.isTrain:
            netD_local = networks.LocalDiscriminator(opt)
            netD_local.apply(networks.weights_init)
            netD_global = networks.GlobalDiscriminator(opt)
            netD_global.apply(networks.weights_init)
        else:
            netD_local = None
            netD_global = None

        return netExtractor, netPredictor, netSampler, netD_local, netD_global

    def load_networks(self):
        # if not self.opt.isTrain or self.opt.continue_train:
        #     self.netExtractor = util.load_network(
        #         self.netExtractor, 'Extractor', self.opt.which_epoch, self.opt)
        #     self.netPredictor = util.load_network(
        #         self.netPredictor, 'Predictor', self.opt.which_epoch, self.opt)
        #     self.netSampler = util.load_network(
        #         self.netSampler, 'Sampler', self.opt.which_epoch, self.opt)
        #     self.text_projection = util.load_network(
        #         self.text_projection, 'projection', self.opt.which_epoch, self.opt)
        #     if self.opt.isTrain:
        #         self.netD_local = util.load_network(
        #             self.netD_local, 'D_local', self.opt.which_epoch, self.opt)
        #         self.netD_global = util.load_network(
        #             self.netD_global, 'D_global', self.opt.which_epoch, self.opt)
        if not self.opt.isTrain or self.opt.continue_train:
            self = util.load_network(self, 'VidPredictionModel', self.opt.which_epoch, self.opt)

    # preprocess the input, such as moving the tensors to GPUs
    # |data|: dictionary of the input data
    def preprocess_input(self, data: dict):
        if self.use_gpu():
            data['input_images'] = data['input_images'].cuda()
            if torch.is_tensor(data['gen_images']):
                data['gen_images'] = data['gen_images'].cuda()
            else:
                data['gen_images'] = None
            if torch.is_tensor(data['gen_segs']):
                data['gen_segs'] = data['gen_segs'].cuda()
            else:
                data['gen_segs'] = None
            if torch.is_tensor(data['text']):
                # T5 process text
                data['text'] = data['text'].cuda()
                text = data['text']
                with torch.no_grad():
                    data['text'] = self.netT5.encoder(data['text'])[0]
                data['text'] = data['text'][torch.arange(data['text'].shape[0]), text.argmax(dim=-1)] @ self.text_projection
                data['text'] = data['text'].unsqueeze(2).unsqueeze(3).unsqueeze(4)
                data['text'] = data['text'].cuda()
            else:
                data['text'] = None

        return data

    def compute_generator_loss(self, input_image, real_image_seq, seg_seq, text):
        G_losses = {}

        generate = self.generate_fake(input_image, text)
        # N * 3 * Sum(F) * H * W
        fake_image_seq = torch.cat(generate['refine'], 2)
        # N * 3 * Sum(F) * H * W
        warp_image_seq = torch.cat(generate['warp'], 2)

        self.generate_fake_image_seq = fake_image_seq.detach()

        local_pred_fake, local_pred_real, global_pred_fake, global_pred_real \
            = self.discriminate(fake_image_seq, real_image_seq, input_image)

        G_losses['GAN_local'] = self.criterionGAN(
            local_pred_fake[-1], True, for_discriminator=False) * self.opt.lambda_local
        G_losses['GAN_global'] = self.criterionGAN(
            global_pred_fake[-1], True, for_discriminator=False) * self.opt.lambda_global

        GAN_Feat_loss = 0.0
        # last output is the final prediction, so we exclude it
        num_intermediate_outputs = len(local_pred_fake) - 1
        for i in range(num_intermediate_outputs):  # for each layer output
            unweighted_loss = self.criterionFeat(
                local_pred_fake[i], local_pred_real[i].detach())
            GAN_Feat_loss += unweighted_loss / num_intermediate_outputs
        num_intermediate_outputs = len(global_pred_fake) - 1
        for i in range(num_intermediate_outputs):  # for each layer output
            unweighted_loss = self.criterionFeat(
                global_pred_fake[i], global_pred_real[i].detach())
            GAN_Feat_loss += unweighted_loss / num_intermediate_outputs
        G_losses['GAN_Feat'] = GAN_Feat_loss * self.opt.lambda_feat

        if not self.opt.no_feature_loss:
            warp_images = torch.cat(torch.chunk(warp_image_seq, warp_image_seq.size(
                2), 2), dim=0).squeeze(2)  # (N * Sum(F)) * 3 * H * W
            fake_images = torch.cat(torch.chunk(fake_image_seq, fake_image_seq.size(
                2), 2), dim=0).squeeze(2)  # (N * Sum(F)) * 3 * H * W
            real_images = torch.cat(torch.chunk(real_image_seq, real_image_seq.size(
                2), 2), dim=0).squeeze(2)  # (N * Sum(F)) * 3 * H * W
            # (2N * Sum(F)) * 3 * H * W
            fake_images = torch.cat([fake_images, warp_images], 0)
            real_images = real_images.repeat(
                2, 1, 1, 1)  # (2N * Sum(F)) * 3 * H * W
            feature_loss = self.criterionFeature(
                fake_images, real_images) * self.opt.lambda_feat
            if not self.opt.use_resnet:
                G_losses['VGG'] = feature_loss
            else:
                G_losses['Resnet'] = feature_loss

        if self.opt.lambda_seg > 0:
            gen_seg = torch.cat(generate['seg'], 2)
            if self.opt.label_class > 0:
                assert seg_seq is not None
                loss_Seg = F.cross_entropy(gen_seg, seg_seq)
            else:
                soft_label = gen_seg.argmax(1)
                loss_Seg = F.cross_entropy(gen_seg, soft_label)
            G_losses['Seg'] = loss_Seg * self.opt.lambda_seg

        gen_flow = torch.cat(generate['flow'], dim=2)
        Matrix_Flow = torch.cat(torch.chunk(gen_flow, gen_flow.size(2), 2), dim=0).squeeze(2)  # (N * Sum(F)) * 2 * H * W
        Avg_Matrix_Flow = F.avg_pool2d(Matrix_Flow, kernel_size=self.opt.kernel_matrix+1,
                                       stride=1, padding=self.opt.kernel_matrix//2, count_include_pad=False)
        loss_intra_patch = self.criterionFeat(Matrix_Flow, Avg_Matrix_Flow)
        loss_intra_line = (self.criterionFeat(Matrix_Flow[:, 0:1], Matrix_Flow[:, 0:1].mean(2, keepdim=True))
                           + self.criterionFeat(Matrix_Flow[:, 1:2], Matrix_Flow[:, 1:2].mean(3, keepdim=True))) * 0.5
        G_losses['intra'] = (loss_intra_patch +
                             loss_intra_line) * self.opt.lambda_intra

        loss_inter = 0.0
        for i in range(self.opt.n_recurrent):
            vid_slice = generate['refine'][i]
            motion_slice = generate['motion'][i]
            content_slice = generate['content'][i]
            content_reextract, motion_reextract = self.netExtractor(
                vid_slice, text,'video')
            # + self.criterionFeat(content_slice, content_reextract)
            loss_inter += self.criterionFeat(motion_slice, motion_reextract)
        G_losses['inter'] = loss_inter * self.opt.lambda_inter

        return G_losses, fake_image_seq.transpose(1, 2)  # N * F * 3 * H * W

    def compute_discriminator_loss(self, input_image, real_image_seq):
        D_losses = {}
        with torch.no_grad():
            # generate= self.generate_fake(input_image)
            fake_image_seq = self.generate_fake_image_seq.detach()
            fake_image_seq.requires_grad_()

        local_pred_fake, local_pred_real, global_pred_fake, global_pred_real \
            = self.discriminate(fake_image_seq, real_image_seq, input_image)

        loss_D_Fake_local = self.criterionGAN(
            local_pred_fake, False, for_discriminator=True)
        loss_D_Real_local = self.criterionGAN(
            local_pred_real, True, for_discriminator=True)

        D_losses['D_Fake_local'] = loss_D_Fake_local
        D_losses['D_real_global'] = loss_D_Real_local

        loss_D_Fake_global = self.criterionGAN(
            global_pred_fake, False, for_discriminator=True)
        loss_D_Real_global = self.criterionGAN(
            global_pred_real, True, for_discriminator=True)

        D_losses['D_Fake_global'] = loss_D_Fake_global
        D_losses['D_real_global'] = loss_D_Real_global

        return D_losses

    def generate_fake(self, input_seq, input_text):
        '''
        para :
            input_seq : Tensor(N * C * F' * H * W)
        '''
        generate = {}

        # stage 1
        vid_content, vid_motion = self.netExtractor(input_seq, input_text, 'video')

        generate['vid_content'] = vid_content
        generate['vid_motion'] = vid_motion

        motion_set = []
        content_set = []
        seg_set = []
        flow_set = []
        warp_set = []
        refine_set = []

        latest_image = input_seq[:, :, -1]

        for i in range(self.opt.n_recurrent):
            # stage 2
            predicted_seg, predicted_flow = self.netPredictor(
                vid_content if i == 0 else None, vid_motion if i == 0 else None, text=input_text, size=input_seq.size()[-2:])
            motion_set.append(self.netPredictor.motion)
            content_set.append(self.netPredictor.content)
            seg_set.append(predicted_seg)
            flow_set.append(predicted_flow)

            warp_seq = []
            for n in range(predicted_flow.size(2)):
                warp_img = self.resample(latest_image, predicted_flow[:, :, n])
                warp_seq += [warp_img.unsqueeze(2)]
            warp_seq = torch.cat(warp_seq, 2)  # N * 3 * F * H * W
            warp_set.append(warp_seq)
            
            # stage 3
            refine_seq = self.netSampler(
                warp_seq, predicted_seg, input_text, size=input_seq.size()[-2:])  # N * 3 * F * H * W
            refine_set.append(refine_seq)

            latest_image = refine_seq[:, :, -1]

        generate['motion'] = motion_set
        generate['content'] = content_set
        generate['seg'] = seg_set
        generate['flow'] = flow_set
        generate['warp'] = warp_set
        generate['refine'] = refine_set

        return generate

    @torch.no_grad()
    def inference(self, _input, _text , _motion=None):
        '''
        para :
            _input : Tensor Video (N * 3 * F * H * W)
        '''
        content, motion = self.netExtractor(_input, _text, input_type='video')
        latest_image = _input[:, :, -1]

        refine_set = []
        for i in range(self.opt.n_recurrent):
            predicted_seg, predicted_flow = self.netPredictor(
                content if i == 0 else None, motion if i == 0 else None, _text, size=_input.size()[-2:])

            warp_seq = []
            for n in range(predicted_flow.size(2)):
                warp_img = self.resample(latest_image, predicted_flow[:, :, n])
                warp_seq += [warp_img.unsqueeze(2)]
            warp_seq = torch.cat(warp_seq, 2)  # N * 3 * F * H * W

            refine_seq = self.netSampler(
                warp_seq, predicted_seg, _text, size=_input.size()[-2:])  # N * 3 * F * H * W
            refine_set.append(refine_seq)

            latest_image = refine_seq[:, :, -1]

        return torch.cat(refine_set, 2).transpose(1, 2)

    def discriminate(self, fake_image_seq, real_image_seq, input_image_seq=None):
        '''
        para :
            fake_image_seq  : N * C * Sum(F) * H * W
            real_image_seq  : N * C * Sum(F) * H * W
            input_image_seq : N * C * F' * H * W
        '''
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image_seq, real_image_seq], dim=0)

        # Content loss
        # [2N * C1 * Sum(F) * H1 * W1, ..., N * 1 * Sum(F) * H' * W']
        content_pred = self.netD_local(fake_and_real)

        content_pred_fake, content_pred_real = self.divide_pred(
            content_pred)

        # Consistence
        if input_image_seq is not None:
            fake_image_seq = torch.cat(
                [input_image_seq, fake_image_seq], dim=2)
            real_image_seq = torch.cat(
                [input_image_seq, real_image_seq], dim=2)
            fake_an_real = torch.cat(
                [fake_image_seq, real_image_seq], dim=0)
        # [2N * C1 * Sum(F)//2 * H1 * W1, ..., N * 1 * Sum(F)//K * H' * W']
        consistence_pred = self.netD_global(fake_and_real)

        consistence_pred_fake, consistence_pred_real = self.divide_pred(
            consistence_pred)

        return content_pred_fake, content_pred_real, consistence_pred_fake, consistence_pred_real

    # Take the prediction of fake and real images from the combined batch

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append(p[:p.size(0) // 2])
                real.append(p[p.size(0) // 2:])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
