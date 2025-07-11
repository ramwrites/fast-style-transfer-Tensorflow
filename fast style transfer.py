
# used for first epoch training
cfg = {
    'style_weight': 5.0,
    'content_weight': 1.0,
    'tv_weight': 1e-5
}
# used for second epoch training
cfg = {
    'style_weight': 1e2,
    'content_weight': 1e5,
    'tv_weight': 1e-6
}


"""### Importing libraries"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras

"""### Loss Net"""

vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False

# vgg.summary()

content_layers = ['block2_conv2']
style_layers=['block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3']

total_layers = style_layers + content_layers


def mini_model(model, layers):
    output = [model.get_layer(layer).output for layer in layers]
    model = tf.keras.models.Model(vgg.input, output)
    return model

vgg_model = mini_model(vgg, total_layers)

"""### Loss Functions"""

def gram_matrix(matrix):
    B, H, W, C = matrix.shape
    gram = tf.linalg.einsum('bhwc,bhwd -> bcd', matrix, matrix)
    return gram/tf.cast(matrix.shape[1] * matrix.shape[2] * matrix.shape[3], dtype=tf.float32)

def style_loss(style_activations, gen_activations):
    s_loss = 0
    for i in range(4):
        style_gram = gram_matrix(style_activations[i])
        gen_gram = gram_matrix(gen_activations[i])

        s_loss += tf.reduce_sum(tf.square(style_gram - gen_gram), axis = [1,2])

    return s_loss

def content_loss(content_activations, gen_activations):
    f_loss = tf.reduce_sum(tf.square(gen_activations - content_activations), axis = [1,2,3])
    return f_loss/tf.cast(content_activations.shape[1]*content_activations.shape[2]*content_activations.shape[3], dtype=tf.float32)

def total_loss(gen_out, style_acti, gen_acti, content_acti):
    # total variation
    tv = tf.image.total_variation(gen_out)

    # feat_loss
    c_loss = content_loss(content_activations=content_acti[-1], gen_activations=gen_acti[-1])

    # style_loss
    s_loss = style_loss(style_activations=style_acti[:-1], gen_activations=gen_acti[:-1])

    # total loss
    total = cfg['content_weight']*c_loss + cfg['style_weight']*s_loss + cfg['tv_weight']*tv

    return tf.reduce_mean(total)

"""### Transform Net"""

class down_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pad, stride, **kwargs):
        super(down_block, self).__init__(**kwargs)
        self.pad_size = pad
        self.stride_size = stride
        self.conv2d = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=(self.stride_size, self.stride_size))
        self.relu = tf.keras.layers.ReLU()
        self.instance = tf.keras.layers.GroupNormalization(groups = -1, axis = -1, epsilon=1e-5)
    def call(self, inp):
        x = tf.pad(inp, [[0,0], [self.pad_size,self.pad_size], [self.pad_size,self.pad_size], [0,0]], mode='REFLECT')
        x = self.conv2d(x)
        x = self.instance(x)
        x = self.relu(x)

        return x

class up_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(up_block, self).__init__(**kwargs)
        self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')
        self.conv2d = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding='same')
        self.relu = tf.keras.layers.ReLU()
        self.instance = tf.keras.layers.GroupNormalization(groups = -1, axis = -1, epsilon=1e-5)
    def call(self,inp):
        x = self.upsample(inp)
        x = self.conv2d(x)
        x = self.instance(x)
        x = self.relu(x)

        return x

class residual_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(residual_block, self).__init__(**kwargs)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters = filters, kernel_size=kernel_size)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters = filters, kernel_size=kernel_size)
        self.relu = tf.keras.layers.ReLU()
        self.instance_1 = tf.keras.layers.GroupNormalization(groups = -1, axis = -1, epsilon=1e-5)
        self.instance_2 = tf.keras.layers.GroupNormalization(groups = -1, axis = -1, epsilon=1e-5)
        self.add = tf.keras.layers.Add()
    def call(self, inp):
        res = inp
        x = tf.pad(inp, [[0,0], [1,1], [1,1], [0,0]], mode='REFLECT')
        x = self.conv2d_1(x)
        x = self.instance_1(x)
        x = self.relu(x)
        x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], mode='REFLECT')
        x = self.conv2d_2(x)
        x = self.instance_2(x)
        x = self.add([x, res])

        return x

class transfer_model(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(transfer_model, self).__init__(**kwargs)
        self.down_1 = down_block(filters=32, kernel_size=(9,9), pad=3, stride=1)
        self.down_2 = down_block(filters=64, kernel_size=(3,3), pad=1, stride=2)
        self.down_3 = down_block(filters=128, kernel_size=(3,3), pad=1, stride=2)
        self.res_1 = residual_block(filters = 128, kernel_size=(3,3))
        self.res_2 = residual_block(filters = 128, kernel_size=(3,3))
        self.res_3 = residual_block(filters = 128, kernel_size=(3,3))
        self.res_4 = residual_block(filters = 128, kernel_size=(3,3))
        self.res_5 = residual_block(filters = 128, kernel_size=(3,3))
        self.up_1 = up_block(filters=64, kernel_size=(3,3))
        self.up_2 = up_block(filters=32, kernel_size=(3,3))
        self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=(9,9))
    def call(self,inp):
        x = self.down_1(inp)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = tf.pad(x, [[0,0], [4,4], [4,4], [0,0]], mode='REFLECT')
        x = self.conv(x)
        x = tf.keras.layers.Activation(lambda t: (tf.nn.tanh(t) + 1) * 127.5)(x)
        return x

"""### Importing Images"""

def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=[256,256])
    img = tf.cast(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis = 0)

    return img

test_content = load_img('/content/drive/MyDrive/Colab Notebooks/fst/willis.jpg')

style_img = load_img('/content/drive/MyDrive/Colab Notebooks/fst/starry_night.jpg')

style_activations = vgg_model(tf.keras.applications.vgg16.preprocess_input(style_img))

# I converted the COCO 2014 train dataset to TFRecords to reduce the size of the overall file

def parse_tfr(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, feature_description)

    image = tf.io.decode_jpeg(example['image'])
    image = tf.cast(image, tf.float32)

    return image

def load_tfr(tfr):
    raw_file = tf.data.TFRecordDataset(tfr)
    parsed_dataset = raw_file.map(parse_tfr)
    dataset = parsed_dataset.batch(batch_size=4)

    return dataset

dataset = load_tfr('/content/drive/MyDrive/Colab Notebooks/fst/images.tfrecord')

"""### Training"""

transform_net = transfer_model()

transform_net(tf.zeros((1,256,256,3)))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3) #  changed the learning rate to 1e-4 after 18000 iterations

def train_func(dataset, model, style_activations, test_content):
    step = 0
    batch_losses = []
    for inp_batch in dataset:
        with tf.GradientTape() as tape:
            gen_out = model(inp_batch)
            processed_gen_out = tf.keras.applications.vgg16.preprocess_input(gen_out)
            gen_activations = vgg_model(processed_gen_out)
            processed_inp = tf.keras.applications.vgg16.preprocess_input(inp_batch)
            inp_activations = vgg_model(processed_inp)

            t_loss = total_loss(gen_out, style_activations, gen_activations, inp_activations)
            batch_losses.append(t_loss)
        grads = tape.gradient(t_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        if step%1000 == 0:
            # seeing the progress of the model
            test_out = model(test_content)
            test_out = test_out/255
            plt.imshow(tf.squeeze(test_out.numpy()))
            plt.show()

            # saving the model
            tf.keras.models.save_model(model,os.path.join('/content/drive/MyDrive/Colab Notebooks/fst/saved', f'model{step}.keras'))
            print(f'model saved as model{step}.keras')
            print(f'loss = {tf.reduce_mean(tf.convert_to_tensor(batch_losses)).numpy()}')

        step += 1
    return tf.reduce_mean(tf.convert_to_tensor(batch_losses)).numpy()

epoch_losses = []

for i in range(2):
  loss = train_func(dataset, transform_net, style_activations, test_content)
  epoch_losses.append(loss)
  tf.keras.models.save_model(transform_net,os.path.join('/content/drive/MyDrive/Colab Notebooks/fst/saved', f'epoch{i}.keras'))
  print(f'epoch loss {loss}')


# load and apply style to any image and plot them
def plot_style_transfered_img(img_path):
  img = tf.io.read_file(img_path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.cast(img, dtype=tf.float32)
  img = tf.expand_dims(img, axis = 0)
  img = transform_net(img)
  img =  img/255
  plt.axis('off')
  plt.imshow(tf.squeeze(img))
  plt.show()

