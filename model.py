!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc, numpy as np, os, sys


def load_img(path_to_img):
    """
    Load image and resize it to max 512 pixels on one side, and 4-dimension.
    Input: path of images
    Output: 
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # [:-1] slices the string to omit the last character (the color channel in the shape)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    # add a new axis to the 1st shape position
    img = img[tf.newaxis, :]
    return img

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


def imshow(image, title=None):
    """Plot the image given"""
    if len(image.shape) > 3:
    # Because we added a tf.newaxis to img, so now we squeeze it to 3d again
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

# define how to get the layers
def vgg_layers(layer_names):
    """
    Input: The names of the layers that you want to obtain from VGG19
    Output: A GAN model of the input(vgg input) and the output generated from the layers
    """
    vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """ 
    The image's content can be described by the means and correlations across each feature map. 
    A Gram matrix takes the outer product of the feature vector and itself each precise location. 
    It then averages the outer product over all locations.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2],tf.float32)
    return result/num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name:value  for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value for style_name, value
                        in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    # tf.add_n: Adds all input tensors element-wise.
    # tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

@tf.function()
def train_step(image):

    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

# Constants

# load the content image and style image from the internet


def ffwd(content_path, style_path, paths_out, device_t='/gpu:0', epochs = 4, steps_per_epoch = 100):

    step = 0
    
    content_path = tf.keras.utils.get_file('content.jpg',content_path)
    style_path = tf.keras.utils.get_file('style.jpg',style_path)
    content_im = load_img(content_path)
    style_im = load_img(style_path)
    
    global content_layers
    content_layers = ['block5_conv1', 'block5_conv2'] 
    global style_layers 
    style_layers = ['block1_conv1','block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    global num_content_layers
    num_content_layers = len(content_layers)
    global num_style_layers 
    num_style_layers = len(style_layers)

    global extractor 
    extractor = StyleContentModel(style_layers, content_layers)
    global style_targets 
    style_targets = extractor(style_im)['style']
    global content_targets 
    content_targets = extractor(content_im)['content']
    
    opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)
    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight=1e8

    
    image = tf.Variable(content_im)
    
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
    output = image.read_value()
    save_img(paths_out, output)

def ffwd_to_img(content_im, style_im, out_path, device='/cpu:0'):
    ffwd(content_im, style_im, paths_out = out_path, epochs = 1, device_t=device)