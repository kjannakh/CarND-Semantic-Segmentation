#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
from PIL import Image
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt

num_classes = 2
image_shape = (160, 576) #(590, 1640) #CULane dataset shape
EPOCHS = 1
BATCH_SIZE = 8
DROPOUT = 0.75

# Specify these directory paths

data_dir = './data'
runs_dir = './runs'
training_dir ='../../CULane/'
vgg_path = './data/vgg'

tf.compat.v1.disable_eager_execution()

#correct_label = tf.compat.v1.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
#learning_rate = tf.compat.v1.placeholder(tf.float32)
#keep_prob = tf.compat.v1.placeholder(tf.float32)

# Check TensorFlow Version
assert LooseVersion(tf.compat.v1.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.compat.v1.__version__)
print('TensorFlow Version: {}'.format(tf.compat.v1.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    # load the model and weights
    model = tf.compat.v1.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    # Get Tensors to be returned from graph
    graph = tf.compat.v1.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3, layer4, layer7
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.compat.v1.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.compat.v1.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.compat.v1.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.compat.v1.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op

#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):

            loss, _ = sess.run([cross_entropy_loss, train_op],
            feed_dict={input_image: X_batch, correct_label: gt_batch,
            keep_prob: keep_prob_value, learning_rate:learning_rate_value})

            total_loss += loss

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()
#tests.test_train_nn(train_nn)


def run():

     # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # A function to get batches
    get_batches_fn = helper.gen_batch_function(training_dir, image_shape)
    
    with tf.compat.v1.Session() as session:
            
        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7, num_classes)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)
        
        # Initialize all variables
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.local_variables_initializer())

        print("Model build successful, starting training")

        # Train the neural network
        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
                train_op, cross_entropy_loss, image_input,
                correct_label, keep_prob, learning_rate)

        # Run the model with the test images and save each painted output image (roads painted green)
        helper.save_inference_samples(runs_dir, training_dir, session, image_shape, logits, keep_prob, image_input)
        
        print("All done!")

from glob import glob
data_folder = '../../CULane'
if __name__ == '__main__':
    #run()
    #num_samples = 52857
    #image_paths = []
    #label_paths = []
    #with open(data_folder + '/list/train_gt.txt') as fp:
    #    for i in range(num_samples):
    #        line = fp.readline().split(' ')
    #        image_paths.append(line[0])
    #        label_paths.append(line[1])
    #images = np.array(image_paths)
    #labels = np.array(label_paths)
    #shuffler = np.random.permutation(len(images))
    #images = images[shuffler]
    #labels = labels[shuffler]
    #label_dict = {image: label for image, label in zip(images, labels)}

    #print(label_paths[0,0])
    #print(label_paths.shape)
    #image_paths = np.array(glob(os.path.join(data_folder, 'driver_23*/*/*.jpg')))
    #labels = np.array(glob(os.path.join(data_folder, 'laneseg_label_w16', 'driver_23*/*/*.png')))
    #for label in labels:
    #    if 
    #print(image_paths.shape, labels.shape)
    #background_color = np.array([255, 0, 0])
    #gt_file = '../../CULane/laneseg_label_w16/driver_23_30frame/05151640_0419.MP4/00000.png'
    #image_file = '../../CULane/driver_23_30frame/05151640_0419.MP4/00000.jpg'
    #image = io.imread(gt_file)
    #print(image.shape)
    #mask = (image != 0)
    #new_image = np.zeros((image.shape[0], image.shape[1], 3))
    #new_image[mask,1] = 1
    #plt.imshow(new_image)
    #plt.show()