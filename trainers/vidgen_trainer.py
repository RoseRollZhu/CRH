import torch
import torch.cuda.amp as amp

from models.VidPredictionModel import VidPredictionModel


class VidGenTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.vidgen_model = VidPredictionModel(opt)
        self.vidgen_model.load_networks()
        self.vidgen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.vidgen_model)
        if not opt.isTrain:
            self.vidgen_model.eval()
        self.vidgen_model = self.vidgen_model.cuda()

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.vidgen_model.create_optimizers(
                opt)
            self.old_lr = opt.lr

        if opt.isTrain:
            self.scaler = amp.GradScaler()
        try:

            self.local_rank = torch.distributed.get_rank()

            self.vidgen_model = torch.nn.parallel.DistributedDataParallel(self.vidgen_model, device_ids=[
                                                                        self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            self.vidgen_model_on_one_gpu = self.vidgen_model.module
        except:
            self.vidgen_model.to('cuda')
        

        self.generated = None
        self.losses_recoder = {}

    def run_generator_one_step(self, data: dict):
        self.optimizer_G.zero_grad()
        with amp.autocast(enabled=self.opt.opt_level == 'O1'):
            g_losses, generated = self.vidgen_model(data, mode='generator')
            g_loss = sum(g_losses.values()).mean()
            
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data: dict):
        if self.optimizer_D is not None:
            self.optimizer_D.zero_grad()
            with amp.autocast(enabled=self.opt.opt_level == 'O1'):
                d_losses = self.vidgen_model(data, mode='discriminator')
                d_loss = sum(d_losses.values()).mean()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
            self.d_losses = d_losses
        else:
            self.d_losses = {}

    def inference(self, data: dict):
        with amp.autocast(enabled=self.opt.opt_level == 'O1'):
            images = self.vidgen_model(data, mode='inference')
        return images

    def record_losses(self):
        latest_losses = self.get_latest_losses()
        for k, v in latest_losses.items():
            if k in self.losses_recoder:
                self.losses_recoder[k] += v
            else:
                self.losses_recoder[k] = v
        if 'record_runtime' in self.losses_recoder:
            self.losses_recoder['record_runtime'] += 1
        else:
            self.losses_recoder['record_runtime'] = 1

    def get_latest_losses(self):
        def check_item(loss: dict):
            for k, v in loss.items():
                if not torch.is_tensor(v):
                    print(k, "is not a tensor with value:", v)
        return {**self.g_losses, **self.d_losses}

    def get_avg_losses(self):
        record_runtime = self.losses_recoder['record_runtime']
        avg_losses = self.record_losses
        del avg_losses['record_runtime']
        for k, v in avg_losses.items():
            avg_losses[k] = v / record_runtime
        self.record_losses = {}
        return avg_losses

    def get_latest_generated(self):
        return self.generated.detach()

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.vidgen_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            new_lr_G = new_lr
            new_lr_D = new_lr

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            self.old_lr = new_lr
            if self.local_rank == 0:
                print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
