"""
author: lzhbrian (https://lzhbrian.me)
date: 2019.5.28
"""

import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.utils as vutils

import model as model_module
import data as data_module

from tqdm import tqdm
from tensorboardX import SummaryWriter


def cal_gradient_penalty(netD, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
    """from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L278
    Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.cuda()
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


l2_loss = nn.MSELoss()


def main():

    # PARAMS
    scale_list = [25, 32, 50, 64, 88, 100, 120, 150, 200, 250]
    lr = 5e-4
    lr_decay = lr / 10
    batch_size = 1
    num_epoch = 1600
    num_epoch_decay = 400
    total_epoch = num_epoch + num_epoch_decay
    multiple = 3 # in each epoch, how many time shall we update G, D

    beta1, beta2 = 0.5, 0.999
    lambda_alpha = 10
    lambda_gp = 0.1

    # fixed first z noise
    fixed_z = torch.randn(batch_size, 3, scale_list[0], scale_list[0]).cuda() # NCHW

    # CKPT DIR
    model_save_dir = 'ckpt/%s' % ('debuggin_fix_std_each_scale/')
    if not os.path.exists(model_save_dir):
        os.system('mkdir %s' % model_save_dir)
    writer = SummaryWriter(model_save_dir)

    # START TRAINING
    model_G_list = []
    model_D_list = []

    # std of noise in each scale, only calc first time of each scale
    std_dict = {}

    for model_idx, scale in enumerate(scale_list):
        print('Processing model_idx=%d, scale=%d' % (model_idx, scale))

        dataloader = data_module.get_dataloader(scale, batch_size=batch_size, multiple=multiple)
        assert len(dataloader) == multiple

        model_G, model_D = model_module.get_model(model_idx)

        if model_idx not in [0, 4, 8]:
            model_G.load_state_dict(model_G_list[-1].state_dict())
            model_D.load_state_dict(model_D_list[-1].state_dict())

        model_G = model_G.cuda()
        model_D = model_D.cuda()

        optimizer_G = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_D = optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, beta2))

        for prev_model_G in model_G_list:
            prev_model_G.eval()

        for epoch_idx in tqdm(range(num_epoch + num_epoch_decay)):

            # adjust lr
            if epoch_idx == num_epoch:
                for g in optimizer_G.param_groups:
                    g['lr'] = lr_decay
                for g in optimizer_D.param_groups:
                    g['lr'] = lr_decay

            # generate noise for 3 updates of G,D
            cur_noise_list = []
            for scale_idx in range(0, model_idx + 1):
                noise = torch.randn(batch_size, 3, scale_list[scale_idx], scale_list[scale_idx]).cuda()  # gaussian noise NCHW
                cur_noise_list.append(noise)

            # normally, there should only be one data per dataloader
            for batch_idx, real_img in enumerate(dataloader):

                real_img = real_img.cuda()

                # rec
                rec_img = fixed_z
                for idx, prev_model_G in enumerate(model_G_list):
                    rec_img = prev_model_G(rec_img, torch.zeros(rec_img.size()).cuda())
                    rec_img = F.interpolate(rec_img, size=(scale_list[idx + 1], scale_list[idx + 1]), mode='bilinear')
                # only calculate std in the first epoch
                if std_dict.get(model_idx, -1) == -1:
                    std = torch.mean(torch.sqrt(l2_loss(rec_img.detach(), real_img)))
                    std_dict[model_idx] = std
                rec_img = model_G(rec_img, torch.zeros(rec_img.size()).cuda())

                # fake
                fake_img = cur_noise_list[0] * std_dict[0]
                for idx, prev_model_G in enumerate(model_G_list):
                    noise = cur_noise_list[idx] * std_dict[idx]
                    fake_img = prev_model_G(fake_img, noise)
                    fake_img = F.interpolate(fake_img, size=(scale_list[idx + 1], scale_list[idx + 1]), mode='bilinear')
                noise = cur_noise_list[-1] * std_dict[model_idx]
                fake_img = model_G(fake_img, noise)

                # update param
                batches_done = model_idx * total_epoch * multiple + epoch_idx * len(dataloader) + batch_idx

                # optimize G
                optimizer_G.zero_grad()
                loss_g_rec = l2_loss(rec_img, real_img)
                loss_g_adv = - model_D(fake_img)
                loss_g = loss_g_adv + lambda_alpha * loss_g_rec
                loss_g.backward()
                optimizer_G.step()

                writer.add_scalar('loss/loss_g_adv', loss_g_adv, batches_done)
                writer.add_scalar('loss/loss_g_rec', loss_g_rec, batches_done)
                writer.add_scalar('loss/loss_g', loss_g, batches_done)

                # optimize D
                optimizer_D.zero_grad()
                loss_d_adv = model_D(fake_img.detach()) - model_D(real_img)
                loss_d_grad_penalty = cal_gradient_penalty(model_D, real_img, fake_img.detach(), type='mixed', constant=1.0, lambda_gp=1)[0]
                loss_d = loss_d_adv + lambda_gp * loss_d_grad_penalty
                loss_d.backward()
                optimizer_D.step()

                writer.add_scalar('loss/loss_d_adv', loss_d_adv, batches_done)
                writer.add_scalar('loss/loss_d_grad_penalty', loss_d_grad_penalty, batches_done)
                writer.add_scalar('loss/loss_d', loss_d, batches_done)

                # other param
                writer.add_scalar('lr/lr', optimizer_G.param_groups[0]['lr'], batches_done)
                writer.add_scalar('std/std', std_dict[model_idx], batches_done)
                writer.add_image('fake', vutils.make_grid(fake_img, normalize=True, scale_each=True), batches_done)
                writer.add_image('rec', vutils.make_grid(rec_img, normalize=True, scale_each=True), batches_done)
                writer.add_image('real', vutils.make_grid(real_img, normalize=True, scale_each=True), batches_done)

        model_G_list.append(model_G)
        model_D_list.append(model_D)

    # SAVE MODEL
    for idx in range(len(scale_list)):
        torch.save(model_G_list[idx], '%s/G_%s.pth' % (model_save_dir, idx))
        torch.save(model_D_list[idx], '%s/D_%s.pth' % (model_save_dir, idx))

    writer.close()


if __name__ == '__main__':
    main()
