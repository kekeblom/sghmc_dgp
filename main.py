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
parser.add_argument('-M', default=64, type=int)
parser.add_argument('--batch-size', default=128, type=int)

flags = parser.parse_args()

(Xtrain, Ytrain), (Xtest, Ytest) = observations.mnist('/tmp/mnist')

mean = Xtrain.mean()
std = Xtrain.std()
Xtrain = (Xtrain - mean) / std
Xtest = (Xtest - mean) / std

def compute_z_inner(X, M, feature_maps_out):
    X = X.reshape(-1, 28, 28, 1)
    filter_matrix = np.zeros((6, 6, 1, feature_maps_out))
    filter_matrix[3, 3, 0, 0] = 1.0
    convolution = tf.nn.conv2d(X, filter_matrix,
            [1, 2, 2, 1],
            "VALID")

    with tf.Session() as sess:
        filtered = sess.run(convolution)

    return conv_utils.cluster_patches(filtered.reshape(-1, 12, 12, feature_maps_out), M, 5)

patches = conv_utils.cluster_patches(Xtrain.reshape(-1, 28, 28, 1), flags.M, 6)

base_kernel = kernels.SquaredExponential(input_dim=6*6, lengthscales=2.0)

conv_layer1 = ConvLayer((28, 28, 1), patch_size=6,
	stride=2, base_kernel=base_kernel, Z=patches, feature_maps_out=flags.feature_maps)

rbf = kernels.SquaredExponential(input_dim=5*5*flags.feature_maps, lengthscales=2.0)
patch_extractor = PatchExtractor(input_size=(12, 12, 10), filter_size=5, feature_maps=10, stride=1)
conv_kernel = ConvKernel(rbf, patch_extractor)

Z = compute_z_inner(Xtrain, flags.M, flags.feature_maps)
last_layer = Layer(conv_kernel, 10, Z)

model = DGP(Xtrain.reshape(Xtrain.shape[0], 784),
        Ytrain.reshape(Ytrain.shape[0], 1),
        layers=[conv_layer1, last_layer],
        likelihood=MultiClass(10),
        minibatch_size=flags.batch_size,
        window_size=100)

TRAIN_ITERATIONS = 10000
for _ in range(TRAIN_ITERATIONS):
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
        prediction = mean.mean(axis=0).argmax(axis=1).reshape(X.shape[0], 1)
        correct += (prediction == Y).sum()
    return correct / Ytest.shape[0]



accuracy = measure_accuracy(model)

print("Model accuracy:", accuracy)

