import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from trainers.vidgen_trainer import VidGenTrainer
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from tqdm import tqdm
opt = TestOptions().parse(save=False)
opt.nThreads = 8
opt.batchSize = 1  # test code only supports batchSize = 1
assert len(opt.gpu_ids) == 1  # test code only supports GPU_num = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

os.environ['MASTER_ADDR'] = opt.master_address
os.environ['MASTER_PORT'] = opt.master_port

torch.backends.cudnn.benchmark = True
# torch.distributed.init_process_group(backend="nccl")
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
device = torch.device("cuda")

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# if local_rank == 0:
dataset_size = len(data_loader)
#     if local_rank == 0:
print('#testing samples = %d' % dataset_size)
#     # visualizer = Visualizer(opt)
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)
if not os.path.exists(os.path.join(opt.results_dir, 'synthesized')):
    os.makedirs(os.path.join(opt.results_dir, 'synthesized'))
if not os.path.exists(os.path.join(opt.results_dir, 'real')):
    os.makedirs(os.path.join(opt.results_dir, 'real'))
if not os.path.exists(os.path.join(opt.results_dir, 'synthesized_gif')):
    os.makedirs(os.path.join(opt.results_dir, 'synthesized_gif'))
if not os.path.exists(os.path.join(opt.results_dir, 'real_gif')):
    os.makedirs(os.path.join(opt.results_dir, 'real_gif'))

trainer = VidGenTrainer(opt)
t_size = (opt.display_width, opt.display_height)

trainer.vidgen_model.netT5.encoder = trainer.vidgen_model.netT5.encoder.cuda()
with tqdm(total=len(dataset)) as progress_bar:
    for i, data in enumerate(dataset):
        progress_bar.update(1)
        if opt.max_test_num > 0:
            if i >= opt.max_test_num:
                break
        img_path = data['path']
        # print('[%5d/%5d] %s' % (i+1, len(dataset), img_path))

        # Training
        # text prosessing
        text_tokens_list = torch.tensor([])
        for text in data['text']:
            text_tokens = trainer.vidgen_model.tokenizer(text, return_tensors="pt").input_ids
            text_tokens = text_tokens.squeeze(0)
            while len(text_tokens) < opt.maxTokenLength:
                text_tokens = torch.cat((text_tokens, torch.tensor([0])), dim=0)
            text_tokens = text_tokens[:opt.maxTokenLength].unsqueeze(0)
            text_tokens_list = torch.cat((text_tokens_list, text_tokens))
        data['text'] = text_tokens_list.int()


        generated = trainer.inference(data)  # N * F * 3 * H * W

        t_size = (opt.display_width, opt.display_height)

        base_name = os.path.basename(img_path[0])
        file_cat = base_name.split('.')[-1]

        if not os.path.exists(os.path.join(opt.results_dir, 'synthesized', str(i))):
            os.mkdir(os.path.join(opt.results_dir, 'synthesized', str(i)))
        if not os.path.exists(os.path.join(opt.results_dir, 'real', str(i))):
            os.mkdir(os.path.join(opt.results_dir, 'real', str(i)))
        for f in range(generated.size(1)):
            file_name = ('%05d' % (f)) + '.' + 'png'

            syn_save_path = os.path.join(
                opt.results_dir, 'synthesized', str(i), file_name)
            rel_save_path = os.path.join(
                opt.results_dir, 'real', str(i), file_name)

            generated_image = util.tensor2im(generated[0, f], size=t_size)
            real_image = util.tensor2im(
                data['gen_images'].transpose(1, 2)[0, f], size=t_size)

            util.save_image(generated_image, syn_save_path)
            util.save_image(real_image, rel_save_path)

        file_name = ('%05d' % (i)) + '.gif'

        syn_save_path = os.path.join(opt.results_dir, 'synthesized_gif', file_name)
        generated_image_set = util.tensor2seq(generated.data[0], size=t_size)
        util.save_gif(generated_image_set, syn_save_path, fps=10)

        rel_save_path = os.path.join(opt.results_dir, 'real_gif', file_name)
        real_image_set = util.tensor2seq(
            data['gen_images'].transpose(1, 2).data[0], size=t_size)
        util.save_gif(real_image_set, rel_save_path, fps=10)

