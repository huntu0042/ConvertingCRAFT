
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input


class vgg16_bn(tf.keras.Model):

  def __init__(self):
    super(vgg16_bn, self).__init__()
    self.VGG_MEAN = [103.939, 116.779, 123.68]

    self.input_layer = tf.keras.layers.Input(shape=(224,224,3))

    # Block 1
    self.conv1_1 =  tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',strides=[1, 1],
                           name='conv1_1')
    self.conv1_2 = tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv1_2')
    self.pool1_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_1')

    # Block 2
    self.conv2_1 = tf.keras.layers.Conv2D(filters=128,
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same',
                                          name='conv2_1')
    self.conv2_2 = tf.keras.layers.Conv2D(filters=128,
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same',
                                          name='conv2_2')
    self.pool2_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_1')

    # Block 3
    self.conv3_1 = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same',
                                          name='conv3_1')
    self.conv3_2 = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same',
                                          name='conv3_2')
    self.conv3_3 = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same',
                                          name='conv3_3')
    self.pool3_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3_1')

    # Block 4
    self.conv4_1 = tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv4_1')
    self.conv4_2 = tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv4_2')
    self.conv4_3 = tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv4_3')
    self.pool4_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4_1')

    # Block 5
    self.conv5_1 = tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv5_1')
    self.conv5_2 = tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv5_2')
    self.conv5_3 = tf.keras.layers.Conv2D(filters=512,
                           kernel_size=(3, 3),
                           activation='relu',
                           padding='same',
                           name='conv5_3')
    self.pool5_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_1')

    self.flatten = tf.keras.layers.Flatten()
    self.fc6 = tf.keras.layers.Dense(units=4096, use_bias=True, name='fc6', activation='relu')
    self.fc7 = tf.keras.layers.Dense(units=4096, use_bias=True, name='fc7', activation='relu')
    self.fc8 = tf.keras.layers.Dense(units=1000, use_bias=True, name='fc8', activation=None)






  def call(self, inputs):
    # define input layer
    # print(self.pool1_1)
    #x = self.input_layer(inputs)
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=inputs)
    bgr = tf.concat(axis=3, values=[blue - self.VGG_MEAN[0], green - self.VGG_MEAN[1], red - self.VGG_MEAN[2]])

    x = self.conv1_1(bgr)
    x = self.conv1_2(x)
    x = self.pool1_1(x)

    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.pool2_1(x)

    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.conv3_3(x)
    x = self.pool3_1(x)

    x = self.conv4_1(x)
    x = self.conv4_2(x)
    x = self.conv4_3(x)
    x = self.pool4_1(x)

    x = self.conv5_1(x)
    x = self.conv5_2(x)
    x = self.conv5_3(x)
    x = self.pool5_1(x)

    x = self.flatten(x)
    x = self.fc6(x)
    x = self.fc7(x)
    x = self.fc8(x)

    #prob = tf.nn.softmax(x)

    return x


if __name__ == '__main__':

  image = Image.open("./data/test.jpg")
  image = image.resize([224, 224])
  image = np.array(image)
  image_data = image.astype(np.float32)

  model = vgg16_bn()
  inputs = tf.random.normal(shape=[1, 224, 224, 3])
  x_1 = model(inputs=inputs)

  #x_1 = model(inputs=np.expand_dims(image_data, 0))


  # Load weighs
  weighs = np.load("./pretrain/vgg16.npy", encoding='latin1',allow_pickle=True).item()

  for layer_name in weighs.keys():
      print(layer_name)
      layer = model.get_layer(layer_name)
      print(type(weighs[layer_name]))
      layer.set_weights(weighs[layer_name])


  print(x_1.shape)
  print("##")
  #print(x_2.shape)
  model.summary()
