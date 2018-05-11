# Image-Classification-with-Deep-CNN
A Deep Convolutional Neural Network with drop out layer architecture to decrease over-fitting and to increase accuracy.

## Requirements
- Python 3.6 and above
- Tensoflow 1.6.0 and above
- NumPy
- Pandas
- Matplotlib
- PIL
- Also be using a [utility library](http://deeplearning.net/tutorial/code/utils.py) to understand the outputs better.

## Data
Using [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for this to test our model architechture's accuracy against other algorithms.
MNIST is a database of handwritten digits that has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

We can import the dataset using TensorFlow built-in feature, as shown below-
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```


