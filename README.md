<img src='imgs/teaser.png', width="800px">
# Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis

Pytorch implementation for our MSGAN (**Miss-GAN**). We propose a simple yet effective mode seeking regularization term that can be applied to **arbitrary** conditional generative adversarial networks in different tasks to alleviate the mode collapse issue and improve the **diversity**.

Contact: Qi Mao (qimao@pku.edu.cn), Hsin-Ying Lee (hlee246@ucmerced.edu), and Hung-Yu Tseng (htseng6@ucmerced.edu) 

## Paper
Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis<br>
[Qi Mao](https://sites.google.com/view/qi-mao/)\*, [Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/)\*, [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/)\*, [Siwei Ma](https://scholar.google.com/citations?user=y3YqlaUAAAAJ&hl=zh-CN), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019 (* equal contribution)

Please cite our paper if you find the code useful for your research.
```
@inproceedings{MSGAN,
  author = {Mao, Qi, Lee, Hsin-Ying and Tseng, Hung-Yu and Ma, Siwei, and Yang, Ming-Hsuan},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  title = {Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis},
  year = {2019}
}
```
## Usage

### Prerequisites
- Python 3.5 or Python 3.6
- Pytorch 0.4.0 and torchvision (https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [Tensorflow](https://www.tensorflow.org/) (for tensorboard usage)

### Install
- Clone this repo:
```
git clone https://github.com/HelenMao/MSGAN.git
```
## Training Examples
###Conditoned on Label
```
cd MSGAN/DCGAN-Mode-Seeking
python train.py
```
