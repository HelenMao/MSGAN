import torch
from options import TestOptions
from model import CDCGAN
from saver import *
import os
import torchvision
import torchvision.transforms as transforms

label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')

    dataset = torchvision.datasets.CIFAR10(opts.dataroot, train=False, download=True, transform= transforms.Compose([
                          transforms.Resize(opts.img_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    # model
    print('\n--- load model ---')

    model = CDCGAN(opts)
    model.eval()
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)


    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # test
    print('\n--- testing ---')
    test_class_images =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    generate_class_images = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for idx1, (img1, label) in enumerate(loader):
        print('{}/{}'.format(idx1, len(loader)))
        # img1 = img1.cuda()
        imgs = []
        names = []
        label_id = label[0].numpy()
        test_class_images[label_id] += 1
        # img_name='img_{}'.format(test_class_images[label_id])
        # save_img(img1, img_name, os.path.join(os.path.join(opts.dataroot,'test'), '{}'.format(label_names[label_id])))

    for idx2 in range(opts.num):
        generate_class_images[label_id] += 1
        with torch.no_grad():
            img = model.test_forward(label)
        imgs.append(img)
        names.append('img_{}'.format(generate_class_images[label_id]))
    save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(label_names[label_id])))
    return

if __name__ == '__main__':
    main()
