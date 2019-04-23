import torch
from options import TrainOptions
from model import CDCGAN
from saver import Saver
import torchvision
import os
import torchvision.transforms as transforms

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # daita loader
    print('\n--- load dataset ---')
    os.makedirs(opts.dataroot, exist_ok=True)

    dataset = torchvision.datasets.CIFAR10(opts.dataroot, train=True, download=True, transform= transforms.Compose([
        transforms.Resize(opts.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = CDCGAN(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')

    for ep in range(ep0, opts.n_ep):
        for it, (images, label) in enumerate(train_loader):
            if images.size(0) != opts.batch_size:
                continue
            # input data
            images = images.cuda(opts.gpu).detach()
            # update model
            model.update_D(images, label)
            model.update_G()

            # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1


        # save result image
        saver.write_img(ep, model)
        # Save network weights
        saver.write_model(ep, total_it, model)
    return

if __name__ == '__main__':
    main()
