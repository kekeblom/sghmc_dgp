import numpy as np
import pandas
from sklearn import cluster
import tensorflow as tf

from models import RegressionModel, ClassificationModel
from sghmc_dgp import DGP, Layer
from layers import ConvLayer
import kernels
from likelihoods import MultiClass

import observations

(Xtrain, Ytrain), (Xtest, Ytest) = observations.mnist('/tmp/mnist')

mean = Xtrain.mean()
std = Xtrain.std()
Xtrain = (Xtrain - mean) / std
Xtest = (Xtest - mean) / std

def _sample_patches(HW_image, N, patch_size, patch_length):
    out = np.zeros((N, patch_length))
    for i in range(N):
        patch_y = np.random.randint(0, HW_image.shape[0] - patch_size)
        patch_x = np.random.randint(0, HW_image.shape[1] - patch_size)
        out[i] = HW_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size].reshape(patch_length)
    return out

def _sample(tensor, count):
    chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
    return tensor[chosen_indices]

def cluster_patches(NHWC_X, M, patch_size):
    NHWC = NHWC_X.shape
    patch_length = patch_size ** 2 * NHWC[3]
    # Randomly sample images and patches.
    patches = np.zeros((M, patch_length), dtype=NHWC_X.dtype)
    patches_per_image = 1
    samples_per_inducing_point = 100
    for i in range(M * samples_per_inducing_point // patches_per_image):
        # Sample a random image, compute the patches and sample some random patches.
        image = _sample(NHWC_X, 1)[0]
        sampled_patches = _sample_patches(image, patches_per_image,
                patch_size, patch_length)
        patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

    k_means = cluster.KMeans(n_clusters=M,
            init='random', n_jobs=-1)
    k_means.fit(patches)
    return k_means.cluster_centers_

def compute_z(X, M):
    X = X.reshape(-1, 28, 28, 1)
    filter_matrix = np.zeros((6, 6, 1, 1))
    sess = tf.Session()
    filter_matrix[3, 3, 0, 0] = 1.0
    convolution = tf.nn.conv2d(X, filter_matrix,
            [1, 2, 2, 1],
            "VALID")

    filtered = sess.run(convolution)
    sess.close()
    kmeans = cluster.KMeans(n_clusters=M, init='random', n_jobs=-1)
    kmeans.fit(filtered.reshape(filtered.shape[0], 144))
    return kmeans.cluster_centers_


patches = cluster_patches(Xtrain.reshape(-1, 28, 28, 1), 32, 6)

base_kernel = kernels.SquaredExponential(input_dim=6*6, lengthscales=2.0)

conv_layer1 = ConvLayer((28, 28, 1), patch_size=6,
	stride=2, base_kernel=base_kernel, Z=patches, feature_maps_out=1)

rbf = kernels.SquaredExponential(input_dim=12*12, lengthscales=2.0)
Z = compute_z(Xtrain, 32)
rbf_layer = Layer(rbf, 10, Z)

model = DGP(Xtrain.reshape(Xtrain.shape[0], 784), Ytrain.reshape(Ytrain.shape[0], 1),
        layers=[conv_layer1, rbf_layer],
        likelihood=MultiClass(10),
        minibatch_size=512,
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

