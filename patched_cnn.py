import tensorflow as tf

class Config:
    def __init__(self, **kwargs):
        self.training = kwargs["train"] # Boolean to check if training
        if self.training:
            self.train_data = kwargs["train_data"]
            self.test_data = kwargs["test_data"]
        else:
            self.data = kwargs["data"]

class Patched_CNN(object):
    """
    Implementation of 'Patched CNN' architecture from
    http://chenpingyu.org/docs/yago_eccv2016.pdf (page 9)

    CNN that accepts 4 channels; Red, Green, Blue and Patch.
    Here, Patch is meant to be a mask which specifies if the pixel is part the
    the segment being classified, or just from a neighbouring segment.
    """
    def __init__(self, config):
        super(Patched_CNN, self).__init__()

        self.training = config["train"] # Boolean to check if training
        if self.training:
            self.train_data = config["train_data"]
            self.test_data = config["test_data"]
        else:
            self.data = kwargs["data"]

    def load_model(self):
        pass

    def build_model(self, features, labels, mode):
        batch_size   = -1     # dynamic batch size.
        image_height = 32
        image_width  = 32
        channels     = 4
        input_layer = tf.reshape(
                features["x"],
                [batch_size, image_height, image_width, channels])

        # 32x32x4 -> conv1 -> 30x30x50
        conv1 = tf.layers.conv2d(
                inputs = input_layer,
                filters=50,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.nn.relu)

        # 30x30x50 -> conv2 -> 28x28x50
        conv2 = tf.layers.conv2d(
                inputs = conv2,
                filters=50,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.nn.relu)

        # 28x28x50 -> pool1 -> 14x14x50
        pool1 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=(2, 2),
                strides=1)

        # 14x14x50 -> conv3 -> 12x12x50
        conv3 = tf.layers.conv2d(
                inputs = pool1,
                filters=50,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.nn.relu)

        # 12x12x50 -> conv4 -> 10x10x50
        conv4 = tf.layers.conv2d(
                inputs = conv3,
                filters=50,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.nn.relu)

        # 10x10x50 -> conv5 -> 10x10x30
        conv5 = tf.layers.conv2d(
                inputs = conv4,
                filters=30,
                kernel_size=(1, 1),
                padding="same",
                activation=tf.nn.relu)

        # 10x10x30 -> pool2 -> 5x5x30
        pool2 = tf.layers.max_pooling2d(
                inputs=conv5,
                pool_size=(2, 2),
                strides=1)

        # 5x5x30 -> conv6 -> 3x3x50
        conv6 = tf.layers.conv2d(
                inputs = pool2,
                filters=50,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.nn.relu)

        # Dense Layer
        # conv6_flat = tf.reshape(pool2, [-1, 3*3*50]) # use this if padding="valid"
        conv6_flat = tf.reshape(pool2, [-1, 32*32*50])
        dense = tf.layers.dense(
                inputs=conv6_flat,
                units=1024,
                activation=tf.nn.relu)

if __name__ == '__main__':
    train_data = []
    test_data = []
    config = {
            "train": True,
            "train_data": train_data,
            "test_data": test_data
            }
    cnn = Patched_CNN(config)

