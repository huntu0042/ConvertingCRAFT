import numpy as np
import tensorflow as tf

class Maploss(tf.keras.losses.Loss):
    def __init__(self, use_gpu = True,name='MapLoss'):
        super().__init__(name=name)
        print("map loss object created..")


    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        # print("loss")
        # print(pre_loss.shape)
        # print(tf.reshape(pre_loss,[-1]))
        # print(tf.reduce_mean(tf.reshape(pre_loss,[-1])))
        sum_loss = tf.reduce_mean(tf.reshape(pre_loss,[-1])) * 0
        pre_loss = tf.reshape(pre_loss,[batch_size,-1])
        loss_label = tf.reshape(loss_label,[batch_size,-1])
        #print(pre_loss.shape)

        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = tf.reduce_mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = tf.reduce_mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    pick = tf.math.top_k(pre_loss[i][(loss_label[i] < 0.1)],3*positive_pixel)[0]
                    nega_loss = tf.reduce_mean(pick)
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                pick = tf.math.top_k(pre_loss[i],500)[0]
                nega_loss = tf.reduce_mean(pick)
                average_number += 500
                sum_loss += nega_loss
            #sum_loss += loss/average_number

        return sum_loss



    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah

        assert p_gh.shape == gh_label.shape and p_gah.shape == gah_label.shape
        loss1 = tf.compat.v1.losses.mean_squared_error(p_gh, gh_label,reduction=tf.compat.v1.losses.Reduction
.NONE)
        loss2 = tf.compat.v1.losses.mean_squared_error(p_gah, gah_label,reduction=tf.compat.v1.losses.Reduction
.NONE)
        loss_g = tf.matmul(loss1, mask)
        loss_a = tf.matmul(loss2, mask)

        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        print("#loss#")
        print(char_loss)
        print(affi_loss)

        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]