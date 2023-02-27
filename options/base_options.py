import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='Img2Vid', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--local_rank', type=int, help='local rank for this process')
        self.parser.add_argument('--master_address', type=str, default='127.0.0.1', help='the master host ip')
        self.parser.add_argument('--master_port', type=str, default='14786', help='the master host port')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='CRH', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--norm_G', type=str, default='spectralspadeinstance3x3', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--opt_level', type=str, default='O1')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images width to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--label_class', type=int, default=17, help='# of input label class')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--text_nc', type=int, default=512, help='# of input text channels')
        self.parser.add_argument('--maxTokenLength', type=int, default=16, help='# max input text tokens')


        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes')
        self.parser.add_argument('--data_load_skip', default=10, type=int, help='# slice frame')
        self.parser.add_argument('--data_load_length_min', default=5, type=int, help='# min loading frame of model input')
        self.parser.add_argument('--data_load_length_max', default=5, type=int, help='# max loading frame of model input')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--n_frame', type=int, default=10, help='# of output image per forward process')
        self.parser.add_argument('--n_recurrent', type=int, default=3, help='# of recurrent times for Generator')
        self.parser.add_argument('--condition_length', type=int, default=1, help='# of input numbers of images')
        self.parser.add_argument('--read_from_npy', type=int, default=0, help='if read from preprocessed npy')
        

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_width', type=int, default=256,  help='display website width')
        self.parser.add_argument('--display_height', type=int, default=128,  help='display website height')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_block', type=int, default=3, help='# of residual blocks in generator')
        self.parser.add_argument('--code_length', type=int, default=128, help='# of noise dim in generator')
        self.parser.add_argument('--n_downsample', type=int, default=4, help='# downsample in generator')
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        # if self.opt.continue_train == 1:
        #     self.opt.continue_train = True
        # else:
        #     self.opt.continue_train = False

        if self.opt.nThreads < 0:
            self.opt.nThreads = 2 * self.opt.batchSize

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            #torch.cuda.set_device(self.opt.gpu_ids[0])
            assert self.opt.batchSize % len(self.opt.gpu_ids) == 0

        if not os.path.exists(self.opt.checkpoints_dir):
            self.opt.checkpoints_dir = self.opt.checkpoints_dir.replace("hhd", "hhd12306")
            self.opt.dataroot = self.opt.dataroot.replace("hhd", "hhd12306")

        args = vars(self.opt)

        # .local_rank == 0:
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
