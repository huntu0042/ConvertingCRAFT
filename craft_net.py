
import tensorflow as tf
import numpy as np
from PIL import Image
from vgg import vgg16_bn


class double_conv(tf.keras.Model):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=mid_ch,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_ch,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    @tf.function
    def call(self, x):
        x = self.conv(x)
        return x


class CRAFT(tf.keras.Model):

    def __init__(self):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn()
        inputs = tf.random.normal(shape=[1, 768, 768, 3])
        x_1 = self.basenet(inputs=inputs)

        # Load weighs
        weighs = np.load("./pretrain/vgg16.npy", encoding='latin1', allow_pickle=True).item()
        for layer_name in weighs.keys():
            try:
                layer = self.basenet.get_layer(layer_name)
                layer.set_weights(weighs[layer_name])
            except Exception as ex:
                print(ex)
        self.basenet.summary()

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same'
                                   ),
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same'
                                   ),
            tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same'
                                   ),
            tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='same'
                                   ),
            tf.keras.layers.Conv2D(filters=num_class,
                                   kernel_size=(1, 1),
                                   activation='relu'
                                   )]
        )

    @tf.function
    def call(self, inputs):
        sources = self.basenet(inputs)

        #print(sources[0].shape)
        #print(sources[1].shape)

        y = tf.concat([sources[0], sources[1]],-1)
        y = self.upconv1(y)
        y = tf.image.resize(y, sources[2].shape[1:3])
        y = tf.concat([y, sources[2]],-1)
        y = self.upconv2(y)

        y = tf.image.resize(y, sources[3].shape[1:3])
        y = tf.concat([y, sources[3]], -1)
        y = self.upconv3(y)

        y = tf.image.resize(y, sources[4].shape[1:3])
        y = tf.concat([y, sources[4]], -1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y, feature



if __name__ == '__main__':

  image = Image.open("./data/test.jpg")
  image = image.resize([224, 224])
  image = np.array(image)
  image_data = image.astype(np.float32)

  model = CRAFT()
  inputs = tf.random.normal(shape=[1, 224, 224, 3])
  x_1 = model(inputs=inputs)

  #x_1 = model(inputs=np.expand_dims(image_data, 0))


  #print(x_2.shape)
  #model.summary()