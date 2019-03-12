import networks
import torch
import torch.nn as nn

class CDCGAN(nn.Module):
    def __init__(self, opts):
        super(CDCGAN, self).__init__()
        # parameters
        lr = 0.0002
        self.nz = opts.nz
        self.class_num = opts.class_num
        self.G = networks.generator(opts)
        self.D = networks.discriminator(opts)

        self.gen_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.dis_opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        self.BCE_loss = torch.nn.BCELoss()

    def initialize(self):
        self.G.weight_init()
        self.D.weight_init()

    def setgpu(self, gpu):
        self.gpu = gpu
        self.D.cuda(self.gpu)
        self.G.cuda(self.gpu)

    def get_z_random(self, batchSize, nz):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        return z

    def onehot_encoding(self, label):
        onehot = torch.zeros(self.class_num, self.class_num)
        index = torch.zeros([self.class_num, 1], dtype= torch.int64)
        for i in range(self.class_num):
            index[i] = i

        onehot = onehot.scatter_(1, index, 1).view(self.class_num, self.class_num, 1, 1)
        label_one_hot = onehot[label]
        return label_one_hot.cuda(self.gpu).detach()

    def forward(self):
        self.label_one_hot = self.onehot_encoding(self.label)
        self.z_random = self.get_z_random(self.real_image.size(0), self.nz)
        self.z_random2 = self.get_z_random(self.real_image.size(0), self.nz)
        z_conc = torch.cat((self.z_random, self.z_random2), 0)
        label_conc = torch.cat((self.label_one_hot, self.label_one_hot),0)
        self.fake_image=self.G.forward(z_conc, label_conc)
        self.fake_image1, self.fake_image2 = torch.split(self.fake_image, self.z_random.size(0), dim=0)
        self.image_display = torch.cat((self.real_image.detach().cpu(), self.fake_image1.cpu(), self.fake_image2.cpu()), dim=2)

    def update_D(self, image, label):
        self.real_image = image
        self.label = label
        self.forward()

        # update discriminator
        self.dis_opt.zero_grad()
        self.loss_D = self.backward_D(self.D, self.real_image, self.fake_image1, self.label_one_hot)+ \
                       self.backward_D(self.D, self.real_image, self.fake_image2, self.label_one_hot)
        self.loss_D.backward()
        self.dis_opt.step()

    def update_G(self):
        self.gen_opt.zero_grad()
        self.loss_G_GAN = self.backward_G(self.D, self.fake_image1, self.label_one_hot)+ \
                        self.backward_G(self.D, self.fake_image2, self.label_one_hot)

        # mode seeking loss
        lz = torch.mean(torch.abs(self.fake_image2 - self.fake_image1)) / torch.mean(
            torch.abs(self.z_random2 - self.z_random))
        eps = 1 * 1e-5
        self.loss_lz = 1 / (lz + eps)

        self.loss_G = self.loss_G_GAN + self.loss_lz
        self.loss_G.backward()
        self.gen_opt.step()

    def backward_D(self, netD, real, fake, label):
        pred_fake = netD.forward(fake.detach(), label)
        pred_real = netD.forward(real, label)
        all0 = torch.zeros_like(pred_fake).cuda(self.gpu)
        all1 = torch.ones_like(pred_real).cuda(self.gpu)
        ad_fake_loss = nn.functional.binary_cross_entropy(pred_fake, all0)
        ad_true_loss = nn.functional.binary_cross_entropy(pred_real, all1)
        loss_D = ad_true_loss + ad_fake_loss
        return loss_D


    def backward_G(self, netD, fake, label):
        pred_fake = netD.forward(fake, label)
        all_ones = torch.ones_like(pred_fake).cuda(self.gpu)
        loss_G = nn.functional.binary_cross_entropy(pred_fake, all_ones)
        return loss_G

    def update_lr(self):
        self.dis_sch.step()
        self.gen_sch.step()

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.D.load_state_dict(checkpoint['dis'])
        self.G.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
                'dis': self.D.state_dict(),
                'gen': self.G.state_dict(),
                'dis_opt': self.dis_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'ep': ep,
                'total_it': total_it
                    }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        image_real = self.normalize_image(self.real_image).detach()
        image_fake1 = self.normalize_image(self.fake_image1).detach()
        image_fake2 = self.normalize_image(self.fake_image2).detach()
        return torch.cat((image_real,image_fake1,image_fake2),2)

    def normalize_image(self, x):
        return x[:,0:3,:,:]

    def test_forward(self, label):
        label_one_hot = self.onehot_encoding(label)
        z_random = self.get_z_random(label.size(0), self.nz, 'gauss')
        outputs = self.G.forward(z_random, label_one_hot)
        return  outputs