import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
        self.parser.add_argument('--img_size', type=int, default=32, help='resized image size for training')
        self.parser.add_argument('--nz', type=int, default=100, help='dimensions of z')
        self.parser.add_argument('--class_num', type=int, default=10, help='class number of the dataset')


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        #data loader related
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')

        #ouptput related
        self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

        # training related
        self.parser.add_argument('--n_ep', type=int, default=200, help='number of epochs')
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        # data loader related
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')

        # ouptput related
        self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
        self.parser.add_argument('--name', type=str, default='CIFAR10', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')

        # model related
        self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt
