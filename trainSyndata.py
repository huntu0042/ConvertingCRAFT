
import tensorflow as tf
import argparse
import time
import random
from data_loader import *

parser = argparse.ArgumentParser(description='CRAFT reimplementation')


parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=16, type = int,
                    help='batch size of training')
#parser.add_argument('--cdua', default=True, type=str2bool,
                    #help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')


def gen():
    for i in range(0,10):
        print(i)
        yield (i, [i] * i)



if __name__ == '__main__':
    # synthtextloader = Synth80k('/home/jiachx/publicdatasets/SynthText/SynthText', target_size=768, viz=True, debug=True)
    # train_loader = torch.utils.data.DataLoader(
    #     synthtextloader,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)
    # train_batch = iter(train_loader)
    # image_origin, target_gaussian_heatmap, target_gaussian_affinity_heatmap, mask = next(train_batch)
    from craft_net import CRAFT

    #net = CRAFT(freeze=True)
    synthtextloader = Synth80k('/Users/yunseong/Documents/dev/dataset/test12', target_size=768, viz=True, debug=True)
    print(synthtextloader)
    print(synthtextloader.pull_item(1))

    dataset = tf.data.Dataset.from_generator(synthtextloader.generate_data,output_types=(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))

    #dataset2 = tf.data.Dataset.from_generator(gen,
    #                                         (tf.int64, tf.int64),
    #                                        (tf.TensorShape([]), tf.TensorShape([None])))
    dataset.batch(batch_size=2)
    iterator = dataset.make_one_shot_iterator()
    #x,y,z,k,f = iterator.get_next()

    print(list(dataset.as_numpy_iterator()))

    #for epoch in range(300):



    # print(x)
    # print(y)
    # print(z)
    # print(k)
    # print(f)

    #print(x.shape, y.shape,z.shape,k.shape,f.shape)

    #net = CRAFT()