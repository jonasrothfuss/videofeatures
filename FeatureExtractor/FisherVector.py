import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class FisherVector:

  def __init__(self, n_kernels=1, covariance_type='diag'):
    assert covariance_type in ['diag', 'full']
    assert n_kernels > 0

    self.n_kernels = n_kernels
    self.covariance_type = covariance_type
    self.fitted = False


  def fit(self, X, Y=None):
    assert X.ndim == 2
    self.feature_dim = X.shape[1]

    # fit GMM and store params of fitted model
    gmm = GaussianMixture(n_components=self.n_kernels, covariance_type=self.covariance_type).fit(X)
    self.covars = gmm.covariances_
    self.means = gmm.means_
    self.weights = gmm.weights_

    # if cov_type is diagonal - make sure that covars holds a diagonal matrix
    if self.covariance_type == 'diag':
      cov_matrices = np.empty(shape=(self.n_kernels, self.covars.shape[1], self.covars.shape[1]))
      for i in range(self.n_kernels):
        cov_matrices[i, :, :] = np.diag(self.covars[i, :])
      self.covars = cov_matrices

    assert self.covars.ndim == 3
    self.fitted = True

    return self

  def predict(self, X):
    """
      Computes Fisher Vectors of provided X
    """
    assert self.fitted, "Model (GMM) must be fitted"
    assert X.shape[1] == self.feature_dim

    #likelihood ratio
    gaussians = [multivariate_normal(mean=self.means[k], cov=self.covars[k]) for k in range(self.n_kernels)]
    likelihood = np.vstack([g.pdf(X) for g in gaussians]).T
    likelihood_ratio = likelihood / likelihood.sum(axis=1)[:, None]

    var = np.diagonal(self.covars, axis1=1, axis2=2)


    norm_dev_from_modes = ((X[:, None, :] - self.means[None, :, :])/ var[None, :, :])

    # mean deviation
    mean_dev = np.multiply(likelihood_ratio[:,:,None], norm_dev_from_modes).mean(axis=0)
    mean_dev = np.multiply(1 / np.sqrt(self.weights[:, None]), mean_dev)

    # covariance deviation
    cov_dev = np.multiply(likelihood_ratio[:,:,None], norm_dev_from_modes**2 - 1).mean(axis=0)
    cov_dev = np.multiply(1 / np.sqrt(2 * self.weights[:, None]), cov_dev)

    fisher_vector = mean_dev.T

    assert fisher_vector.ndim == 2 and fisher_vector.shape[1] == 2 * self.n_kernels
    assert fisher.

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
  return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
  return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
  return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
  v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
  return v / np.sqrt(np.dot(v, v))

def likelihood_moment(x, ytk, moment):
  x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
  return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
  gaussians, s0, s1, s2 = {}, {}, {}, {}
  samples = zip(range(0, len(samples)), samples)

  g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
  for index, x in samples:
    gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

  for k in range(0, len(weights)):
    s0[k], s1[k], s2[k] = 0, 0, 0
    for index, x in samples:
      probabilities = np.multiply(gaussians[index], weights)
      probabilities = probabilities / np.sum(probabilities)
      s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
      s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
      s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

  return s0, s1, s2

def fisher_vector(samples, means, covs, w):
  s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
  T = samples.shape[0]
  covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
  a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
  b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
  c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
  fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
  fv = normalize(fv)
  return fv


if __name__ == "__main__":
  dataset = np.random.normal(loc=[3, 5, -3], size=(200, 3))
  dataset = np.concatenate([np.random.normal(loc=[4, -2, 4], size=(80, 3)),dataset], axis=0)
  print(dataset.shape)
  fv = FisherVector(n_kernels=2).fit(dataset)
  fv.predict(dataset)
