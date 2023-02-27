from collections import OrderedDict
from distutils.spawn import spawn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from trainers.vidgen_trainer import VidGenTrainer
import util.util as util
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
import os
import math
import random
import numpy as np
import torch
import torch.multiprocessing as mp
# spawn
from tensorboardX import SummaryWriter

def worker(rank, opt):
    local_rank = rank
    torch.backends.cudnn.benchmark = True
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=len(opt.gpu_ids))
    # local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # trainer = Img2VidTrainer(opt)
    trainer = VidGenTrainer(opt)
    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    iter_counter = IterationCounter(opt, len(data_loader))
    t_size = (opt.display_width, opt.display_height)

    if local_rank == 0:
        visualizer = Visualizer(opt)

    if local_rank == 0:
        logger = SummaryWriter('log')
    if local_rank == 0:
        print('Start Training')
    total_iters = 0
    for epoch in iter_counter.training_epochs():
        data_loader.sampler.set_epoch(epoch)
        if local_rank == 0:
            iter_counter.record_epoch_start(epoch)

        for i, data in enumerate(dataset, start=iter_counter.epoch_iter):
            total_iters += 1
            if local_rank == 0:
                iter_counter.record_one_iteration()

            # Training
            # text prosessing
            text_tokens_list = torch.tensor([])
            for text in data['text']:
                text_tokens = trainer.vidgen_model.module.tokenizer(text, return_tensors="pt").input_ids
                text_tokens = text_tokens.squeeze(0)
                while len(text_tokens) < opt.maxTokenLength:
                    text_tokens = torch.cat((text_tokens, torch.tensor([0])), dim=0)
                text_tokens = text_tokens[:opt.maxTokenLength].unsqueeze(0)
                text_tokens_list = torch.cat((text_tokens_list, text_tokens))
            data['text'] = text_tokens_list.int()
            
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data)

            # train discriminator
            trainer.run_discriminator_one_step(data)

            # Visualizations
            if iter_counter.needs_printing() and local_rank == 0:
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter, logger, total_iters)
                visualizer.plot_current_errors(
                    losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying() and local_rank == 0:
                n_sample = random.randint(0, opt.n_frame * opt.n_recurrent - 1)
                visuals = OrderedDict()
                visuals['input_image'] = util.tensor2im(
                    data['input_images'][0, :, -1], size=t_size)
                visuals['syn_vid'] = util.tensor2seq(
                    trainer.get_latest_generated()[0], size=t_size)
                visuals['real_vid'] = util.tensor2seq(
                    data['gen_images'][0].transpose(0, 1), size=t_size)
                visuals['syn_total_vid'] = util.tensor2seq(
                    torch.cat([data['input_images'].transpose(
                        1, 2), trainer.get_latest_generated().cpu()], 1)[0], size=t_size
                )
                visuals['real_total_vid'] = util.tensor2seq(
                    torch.cat([data['input_images'].transpose(
                        1, 2), data['gen_images'].transpose(1, 2)], 1)[0], size=t_size
                )
                visualizer.display_current_results(
                    visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving() and local_rank == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        if local_rank == 0:
            # end of epoch
            iter_counter.record_epoch_end()

        # save model for this epoch
        if (epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs) and local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

        # linearly decay learning rate after certain iterations
        trainer.update_learning_rate(epoch)

    if local_rank == 0:
        print('Training was successfully finished.')



if __name__ == '__main__':
    opt = TrainOptions().parse()
    os.environ['MASTER_ADDR'] = opt.master_address
    os.environ['MASTER_PORT'] = opt.master_port

    mp.spawn(worker, nprocs=len(opt.gpu_ids), args=(opt, ))
    