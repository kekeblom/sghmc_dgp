import numpy as np
import pandas
from sklearn import cluster
import tensorflow as tf

from models import RegressionModel, ClassificationModel
from sghmc_dgp import DGP, Layer
from conv.layer import ConvLayer, PatchExtractor
from conv.kernels import ConvKernel
import kernels
from likelihoods import MultiClass
from conv import utils as conv_utils

import argparse
import observations

parser = argparse.ArgumentParser()
parser.add_argument('--feature_maps', default=10, type=int)
parser.add_argument('-M', default=128, type=int)
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--iterations', default=30000, type=int)
parser.add_argument('--cifar', action='store_true')
parser.add_argument('--layers', default=3, type=int)

flags = parser.parse_args()

def load_data():
    if flags.cifar:
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.cifar10('/tmp/cifar')
        Xtrain = np.transpose(Xtrain, [0, 2, 3, 1])
        Xtest = np.transpose(Xtest, [0, 2, 3, 1])
        mean = Xtrain.mean((0, 1, 2))
        std = Xtrain.std((0, 1, 2))
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
    else:
        (Xtrain, Ytrain), (Xtest, Ytest) = observations.mnist('/tmp/mnist')
        mean = Xtrain.mean()
        std = Xtrain.std()
        Xtrain = (Xtrain - mean) / std
        Xtest = (Xtest - mean) / std
    return (Xtrain, Ytrain), (Xtest, Ytest)

(Xtrain, Ytrain), (Xtest, Ytest) = load_data()


def compute_z_inner(X, M, feature_maps_out):
    filter_matrix = np.zeros((5, 5, X.shape[3], feature_maps_out))
    filter_matrix[2, 2, :, :] = 1.0
    convolution = tf.nn.conv2d(X, filter_matrix,
            [1, 2, 2, 1],
            "VALID")

    with tf.Session() as sess:
        filtered = sess.run(convolution)

    return conv_utils.cluster_patches(filtered, M, 5)


layers = []
input_size = Xtrain.shape[1:]

Z_inner = compute_z_inner(Xtrain, flags.M, flags.feature_maps)
patches = conv_utils.cluster_patches(Xtrain, flags.M, 5)
for layer in range(0, flags.layers-1):
    if layer == 0:
        stride = 2
        Z = patches
    else:
        stride = 1
        Z = Z_inner

    base_kernel = kernels.SquaredExponential(input_dim=5*5*input_size[2], lengthscales=2.0)

    conv_layer = ConvLayer(input_size, patch_size=5, stride=stride, base_kernel=base_kernel, Z=Z, feature_maps_out=flags.feature_maps)
    layers.append(conv_layer)

    input_size = (conv_layer.patch_extractor.out_image_height, conv_layer.patch_extractor.out_image_width, flags.feature_maps)

rbf = kernels.SquaredExponential(input_dim=5*5*flags.feature_maps, lengthscales=2.0)
patch_extractor = PatchExtractor(input_size, filter_size=5, feature_maps=10, stride=1)
conv_kernel = ConvKernel(rbf, patch_extractor)

layers.append(Layer(conv_kernel, 10, Z))

model = DGP(Xtrain.reshape(Xtrain.shape[0], np.prod(Xtrain.shape[1:])),
        Ytrain.reshape(Ytrain.shape[0], 1),
        layers=layers,
        likelihood=MultiClass(10),
        minibatch_size=flags.batch_size,
        window_size=100,
        adam_lr=0.01)

for _ in range(flags.iterations):
    model.sghmc_step()
    model.train_hypers()
    if _ % 100 == 1:
        print("Iteration {}".format(_))
        model.print_sample_performance()

POSTERIOR_SAMPLES = 100
model.collect_samples(POSTERIOR_SAMPLES, 100)

def measure_accuracy(model):
    batch_size = 128
    batches = Xtest.shape[0] // batch_size
    correct = 0
    for i in range(batches + 1):
        slicer = slice(i * batch_size, (i+1) * batch_size)
        X = Xtest[slicer]
        Y = Ytest[slicer]
        mean, var = model.predict_y(X, POSTERIOR_SAMPLES)
        prediction = mean.mean(axis=0).argmax(axis=1)
        correct += (prediction == Y).sum()
    return correct / Ytest.shape[0]

accuracy = measure_accuracy(model)

print("Model accuracy:", accuracy)

