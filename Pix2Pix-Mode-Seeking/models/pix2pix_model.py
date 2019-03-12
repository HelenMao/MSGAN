import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G','G_GAN', 'G_L1', 'D', 'lz']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'real_B', 'fake_B_random1', 'fake_B_random2']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0

        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample)

        D_output_nc = opt.input_nc + opt.output_nc

        use_sigmoid = opt.gan_mode == 'dcgan'
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get random z
        self.z_random1 = self.get_z_random(self.real_A.size(0), self.opt.nz)
        self.z_random2 = self.get_z_random(self.real_A.size(0), self.opt.nz)

        fake_B = self.netG(torch.cat((self.real_A, self.real_A), 0), torch.cat((self.z_random1, self.z_random2), 0))
        self.fake_B_random1, self.fake_B_random2 = torch.split(fake_B, self.z_random1.size(0), dim=0)

        self.fake_B_random1_condition = torch.cat((self.real_A, self.fake_B_random1), 1)
        self.fake_B_random2_condition = torch.cat((self.real_A, self.fake_B_random2), 1)

        self.real_B_condition = torch.cat((self.real_A, self.real_B), 1)



    def backward_D(self, netD, real, fake1, fake2):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake1 = netD(fake1.detach())
        pred_fake2 = netD(fake2.detach())
        # real
        pred_real = netD(real)
        loss_D_fake1, _ = self.criterionGAN(pred_fake1, False)
        loss_D_fake2, _ = self.criterionGAN(pred_fake2, False)
        loss_D_real1, _ = self.criterionGAN(pred_real, True)
        loss_D_real2, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake1 + loss_D_fake2 + loss_D_real1 + loss_D_real2
        loss_D.backward()
        return loss_D, [loss_D_fake1, loss_D_fake2, loss_D_real1, loss_D_real2]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_G(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_B_random1_condition, self.netD, self.opt.lambda_GAN) +\
        self.backward_G_GAN(self.fake_B_random2_condition, self.netD, self.opt.lambda_GAN)


        # 2, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_random1, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        #3, modes seeking loss
        lz = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random1)) / torch.mean(torch.abs(self.z_random2 - self.z_random1))
        eps = 1 * 1e-5
        loss_lz = 1 / (lz + eps)
        self.loss_lz = loss_lz

        self.loss_G = self.loss_G_GAN + self.loss_lz+ self.loss_G_L1
        self.loss_G.backward()

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_B_condition, self.fake_B_random1_condition, self.fake_B_random2_condition)

            self.optimizer_D.step()

    def update_G(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()
