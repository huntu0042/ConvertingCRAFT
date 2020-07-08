
import tensorflow as tf
import tensorflow_addons as tfa

import argparse
import time, datetime
import random
from data_loader import *


###import file#######
from mseloss import Maploss
from test import test

tf.debugging.set_log_device_placement(True)

parser = argparse.ArgumentParser(description='CRAFT reimplementation')


parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=4, type = int,
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

args = parser.parse_args()

def gen():
    for i in range(0,10):
        print(i)
        yield (i, [i] * i)

class controlLR():
    def __init__(self):
        self.step = 0
        print("LR Object created..")

    def adjustStep(self,step):
        self.step = step

    def adjustLR(self):
        if self.step == 0:
            lr = args.lr
        else:
            lr = args.lr * (0.8 ** step)
        return lr


if __name__ == '__main__':
    from craft_net import CRAFT

    ###data load
    synthtextloader = Synth80k('/home/motion2ai/Desktop/Dev/ocr/dataset/SynthText/SynthText', target_size=768, viz=False, debug=True)
    len_data = len(synthtextloader)


    batch_size = args.batch_size
    dataset = tf.data.Dataset.from_generator(synthtextloader.generate_data,output_types=(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.shuffle(1000,reshuffle_each_iteration=True)
    #it = iter(dataset)

    ###for summary
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    model = CRAFT()
    # 정확도에 대한 Metric의 인스턴스를 만듭니다
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # Optimizer의 인스턴스를 만듭니다
    objectLR = controlLR()
    optimizer = tfa.optimizers.AdamW(learning_rate=objectLR.adjustLR,weight_decay=args.weight_decay)
    lossObject = Maploss()
    print(objectLR.adjustLR())

    def compute_loss(gh_label, gah_label, out1, out2, mask):
        loss_value = lossObject.forward(gh_label, gah_label, out1, out2, mask)
        return loss_value

    loss_save = 0
    st = time.time()
    step_index = 0

    for epoch in range(300):
        # GradientTape 열어줍니다
        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            objectLR.adjustStep(step_index)


        for step, (image, gh_label, gah_label, mask, _) in enumerate(dataset):
            #image = tf.expand_dims(image, 0)

            with tf.GradientTape() as tape:
                # 순방향 전파(forward)를 수행합니다
                out, _ = model(image)
                # print(out.shape)
                out1 = out[:, :, :, 0]
                out2 = out[:, :, :, 1]
                # cv2.imshow("image", image[0].numpy())
                # cv2.imshow("gh_label",gh_label[0].numpy())
                # cv2.imshow("gah_label", gah_label[0].numpy())
                # cv2.imshow("out1", out1[0].numpy())
                # cv2.imshow("out2", out2[0].numpy())
                # cv2.imshow("mask", mask[0].numpy())
                # cv2.waitKey(0)

                loss_value = compute_loss(gh_label, gah_label, out1, out2, mask)

            gradients = tape.gradient(loss_value,model.trainable_weights)
            optimizer.apply_gradients(zip(gradients,model.trainable_weights))

            '''record'''
            loss_save += loss_value
            if step % 2 == 0 and step > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(epoch, step, int(len_data/batch_size), et-st, loss_value/2))
                print(objectLR.adjustLR())
                loss_time = 0
                loss_save = 0
                st = time.time()
            #accuracy.update_state(image,out)

            if step % 10 == 0 :
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value/2, step=step)
                    tf.summary.scalar('learning rate', objectLR.adjustLR(), step=step)

            if step % 5000 == 0 and step != 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value/2, step=step)
                model.save_weights('./checkpoints/my_checkpoint_'+str(step))
                test("checkpoints/my_checkpoint_"+str(step), "test_result/", 0)


    # print(x)
    # print(y)
    # print(z)
    # print(k)
    # print(f)

    #print(x.shape, y.shape,z.shape,k.shape,f.shape)

