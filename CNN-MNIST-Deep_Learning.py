import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from utils import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image

# Initialize an Interactive Session
sess = tf.InteractiveSession()

# Import MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Initialize Parameters
width = 28  # width of image in pixels
height = 28  # height of image in pixels
size = width * height  # number of pixels in image
class_output = 10  # Number of possible classifications for the problem

# Input and Output place holders
x = tf.placeholder(tf.float32, shape=[None, size])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

# Converting images of the dataset to tensors
x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch_number, width, height, image_channel], 1-channel is grayscale
print(x_image)

# Convolutional Layer 1
# Defining kernel weight and bias, 5x5 filter
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  # [width, height, channel, featuremap/filter]
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # 32 biases for 32 outputs


# Convolve with weight and bias tensors
convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

# ReLU activation function
r_conv1 = tf.nn.relu(convolve1)

# Max Pooling of 2v2 matrix with stride of 2 pixels
conv1 = tf.nn.max_pool(r_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(conv1)

# Convolutional Layer 2
# Weights and Biases of Kernels

"""
Input Image from Convolutional Layer-[14, 14, 32], Filter-[5x5x32]. So, 64 filters of size [5x5x32], and the output 
of the convolutional layer would be 64 convolved image, [14x14x64]
"""

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # need 64 biases for 64 outputs

# Convolve with weights and bias tensors
convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

# Apply ReLU activation Function
r_conv2 = tf.nn.relu(convolve2)

# Apply max Pooling
conv2 = tf.nn.max_pool(r_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(conv2)

"""Fully Connected Layer
A fully connected layer uses the Softmax and creates the probabilities. Fully connected layers take all 64 matrices, and
 convert them to a size array.

Each matrix [7x7] will be converted to a matrix of [49x1], and then all of the 64 matrix will be connected, which make 
an array of size [3136x1]. Connect it into another layer of size [1024x1] so, the weight between these 2 layers will be 
[3136x1024]"""

# Flatten Second Layer
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

# Weights and Biases between layer 2 and 3
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

# Applying weights and biases
fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1
print(fc1)

# Apply ReLU activation function
h_fc1 = tf.nn.relu(fc1)
print(h_fc1)

# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
print(layer_drop)

# Readout Layer(Softmax Layer)
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

# Apply weights and biases
fc = tf.matmul(layer_drop, W_fc2) + b_fc2

# Apply Softmax activation Function
y_CNN = tf.nn.softmax(fc)
print(y_CNN)

# Defining functions and training model
# Loss Function, comparing output and layer4 of the tensor
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

# Define Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Define Prediction
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))

# Define accuracy, report accuracy using average of corrected cases
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run Session, train
sess.run(tf.global_variables_initializer())
for i in range(30000):
    batch = mnist.train.next_batch(50)

    # for feedback every few steps
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                  y_: batch[1],
                                                  keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0],
                              y_: batch[1],
                              keep_prob: 0.5})

# Evaluation of model
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images,
                                                    y_: mnist.test.labels,
                                                    keep_prob: 1.0}))

# Visualize the model
kernels = sess.run(tf.reshape(tf.transpose(W_conv1,
                                           perm=[2, 3, 0, 1]), [32, -1]))
image = Image.fromarray(tile_raster_images(kernels,
                                           img_shape=(5, 5),
                                           tile_shape=(4, 8),
                                           tile_spacing=(1, 1)))

# Plot Image
plt.figure(figsize=(20, 20))
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')

# The output of an image passing through first convolution layer
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage, [28, 28]))

# Activated Units
ActivatedUnits = sess.run(convolve1,
                          feed_dict={x: np.reshape(sampleimage,
                                                   [1, 784],
                                                   order='F'),
                                     keep_prob: 1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0, :, :, i], interpolation="nearest", cmap="gray")
plt.show()

# Second convolution layer Images
ActivatedUnits = sess.run(convolve2, feed_dict={x: np.reshape(sampleimage, [1, 784], order='F'), keep_prob: 1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0, :, :, i], interpolation="nearest", cmap="gray")
plt.show()
