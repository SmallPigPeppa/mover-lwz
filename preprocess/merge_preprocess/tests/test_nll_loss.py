import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from pare.losses.uncertainty import MultivariateGaussianNegativeLogLikelihood


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """

    n_dims = int(int(ypreds.shape[1]) / 2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5 * K.sum(K.square((ytrue - mu) / K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return K.mean(-log_likelihood)


def gaussian_nll_np(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """

    n_dims = int(int(ypreds.shape[1]) / 2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:] ** 2

    mse = -0.5 * np.sum(np.square((ytrue - mu) / np.exp(logsigma)), axis=1)
    sigma_trace = -np.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi
    total_loss = np.mean(-log_likelihood)
    print(f'\nMSE: {mse.mean():.2f}'
          f' Sigma: {sigma_trace.mean():.2f}'
          f' log2pi:{log2pi.mean():.2f}'
          f' Total:{total_loss:.2f}')

    return total_loss


if __name__ == '__main__':

    loss = MultivariateGaussianNegativeLogLikelihood()

    mean = np.zeros((1,72))
    sigma = np.ones((1,72))
    ypred = np.concatenate([mean,sigma], axis=1)
    # ytrue = np.zeros((1,72))
    ytrue = np.random.randn(1,72)

    print('YTRUE:', ytrue.shape, ytrue)
    print('YPRED:', ypred.shape, ypred)
    # output = gaussian_nll(ytrue=K.eval(ytrue), ypreds=K.eval(ypred))
    # print('keras', output)

    output = gaussian_nll_np(ytrue=ytrue, ypreds=ypred)
    print('numpy', output)

    output = loss(pred=torch.from_numpy(ypred), gt=torch.from_numpy(ytrue))
    print('torch', output)