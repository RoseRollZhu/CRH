import ast
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=100, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for Generator adam')
        self.parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        # for discriminators
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--D_n_layers', type=int, default=3, help='# of downsample layers in LocalDiscriminator')

        self.parser.add_argument('--lambda_local', type=float, default=1.0, help='weight for local coherent loss')
        self.parser.add_argument('--lambda_global', type=float, default=1.0, help='weight for global coherent loss')
        self.parser.add_argument('--lambda_intra', type=float, default=1.0, help='weight for intra coherent loss')
        self.parser.add_argument('--lambda_inter', type=float, default=10.0, help='weight for inter coherent loss')
        self.parser.add_argument('--lambda_seg', type=float, default=10.0, help='weight for segmentation loss') ########
        self.parser.add_argument('--kernel_matrix', type=int, default=4, help='# of kernel size of flow matrix avgpool')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')

        self.parser.add_argument('--no_feature_loss', action='store_true', help='if specified, do *not* use feature matching loss')
        self.parser.add_argument('--use_resnet', action='store_true', help='if specified, use Resnet feature matching loss')
        self.parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = True
        