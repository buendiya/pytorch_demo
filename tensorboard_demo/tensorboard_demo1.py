# -*-coding:utf-8-*-
import sys
import os
print(os.environ['PYTHONPATH'])
from tensorboard_demo.model import *
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


def write_to_tensorboard():
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)


def inspect_the_model():
    writer.add_graph(net, images)
    writer.close()


if __name__ == '__main__':
    inspect_the_model()
    pass
