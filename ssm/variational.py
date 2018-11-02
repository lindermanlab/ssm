import autograd.numpy as np
import autograd.numpy.random as npr

from ssm.core import _SwitchingLDS
from ssm.emissions import _LinearEmissions
from ssm.preprocessing import interpolate_data

class VariationalPosterior(object):
	"""
	Base class for a variational posterior distribution.
	
		q(z; phi) \approx p(z | x, theta)

	where z is a latent variable and x is the observed data. 

	## Reparameterization Gradients
	We assume that the variational posterior is "reparameterizable"
	in the sense that,

	z ~ q(z; phi)  =d  eps ~ r(eps); z = f(eps; phi).

	where =d denotes equal in distirbution.  If this is the case,
	we can rewrite 

	L(phi) = E_q(z; phi) [g(z)] = E_r(eps) [g(f(eps; phi))] 

	and 

	dL/dphi = E_r(eps) [d/dphi g(f(eps; phi))] 
	        approx 1/S sum_s [d/dphi g(f(eps_s; phi))]  

	where eps_s ~iid r(eps).  In practice, this Monte Carlo estimate
	of dL/dphi is lower variance than alternative approaches like
	the score function estimator.

	## Amortization
	We also allow for "amortized variational inference," in which the
	variational posterior parameters are a function of the data.  We 
	write the posterior as 

		q(z; x, phi) approx p(z | x, theta).
	
	
	## Requirements
	A variational posterior must support sampling and point-wise 
	evaluation in order to be used for the reparameterization trick.
	"""
	def __init__(self, model, datas, inputs=None, masks=None, tags=None):
		"""
		Initialize the posterior with a ref to the model and datas,
		where datas is a list of data arrays.
		"""
		self.model = model
		self.datas = datas

	@property
	def params(self):
		"""
		Return phi.
		"""
		raise NotImplemented

	def sample(self):
		"""
		Return a sample from q(z; x, phi)
		"""
		raise NotImplemented

	def log_density(self, sample):
		"""
		Return log q(z; x, phi)
		"""
		raise NotImplemented


def _initialize_linear_emissions_mean(model, data):
	# y = Cx + d + noise; C orthogonal.  
	# xhat = (C^T C)^{-1} C^T (y-d)
	T = data.shape[0]
	C, d = self.model.emissions.Cs[0], self.model.emissions.ds[0]
	C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

	# TODO: We don't want to project data, we want to project the 
	# data after it's been passed back through the link function. 
	# E.g. for Poisson data with exp link, we want to model log(data)

	if not np.all(mask):
		data = interpolate_data(data, mask)        
	    # We would like to find the PCA coordinates in the face of missing data
	    # To do so, alternate between running PCA and imputing the missing entries
	    for itr in range(25):
	        q_mu = (data-d).dot(C_pseudoinv)
	        data[:, ~mask[0]] = (q_mu.dot(C.T) + d)[:, ~mask[0]]

	# Project data to get the mean
	return (data-d).dot(C_pseudoinv)
	

class SLDSMeanFieldVariationalPosterior(VariationalPosterior):
	"""
	Mean field variational posterior for the continuous latent 
	states of an SLDS.  
	"""
	def __init__(self, model, datas, inputs=None, masks=None, tags=None):
		super(SLDSMeanFieldVariationalPosterior, self).\
			__init__(model, datas, masks, tags)

		# Initialize the parameters
		assert isinstance(model, _SwitchingLDS)
		self.D = model.D
		self.Ts = [data.shape[0] for data in datas]
		self._params = [_initialize_variational_params(data, input, mask, tag)
						for data, input, mask, tag in zip(datas, inputs, masks, tags)]

	@property
	def params(self):
		return self._params

	@params.setter
	def params(self, value):
		assert len(value) == len(self.datas)
		for v, T in zip(value, self.Ts):
			assert len(v) == 2
			q_mu, q_sigma_inv = v
			assert q_mu.shape == q_sigma_inv.shape == (T, D)

		self._params = value

 	def _initialize_variational_params(self, data, input, mask, tag):
 		T = data.shape[0]
 		if isinstance(self.model, _LinearEmissions):
			q_mu = _initialize_linear_emissions_mean(model, data)
			q_sigma_inv = np.zeros((T, self.D))

		else:
			q_mu = np.zeros((T, self.D))
			q_sigma_inv = np.zeros((T, self.D))

		return q_mu, q_sigma_inv

	def sample(self):
		return [q_mu + np.sqrt(np.exp(q_sigma_inv)) * npr.randn(*q_mu.shape) 
				for (q_mu, q_sigma_inv) in self.params]

	def log_density(self, sample):
		assert isinstance(sample, list) and len(sample) == self.len(datas)

		logq = 0
		for s, (q_mu, q_sigma_inv) in zip(sample, self.params):
			assert s.shape == q_mu.shape
			q_sigma = np.exp(q_sigma_inv)
			logq += np.sum(-0.5 * np.log(2 * np.pi * q_sigma))
			logq += np.sum(-0.5 * (s - q_mu)**2 / q_sigma)

		return logq


class SLDSTriDiagVariationalPosterior(VariationalPosterior):
	"""
	Gaussian variational posterior for the continuous latent 
	states of an SLDS.  The Gaussian is constrained to have 
	a block tri-diagonal inverse covariance matrix, as in a 
	linear dynamical system.
	"""
	def __init__(self, model, datas, inputs=None, masks=None, tags=None):
		super(SLDSTriDiagVariationalPosterior, self).\
			__init__(model, datas, masks, tags)

		# Initialize the parameters
		assert isinstance(model, _SwitchingLDS)
		self.D = model.D
		self.Ts = [data.shape[0] for data in datas]
		self._params = [_initialize_variational_params(data, input, mask, tag)
						for data, input, mask, tag in zip(datas, inputs, masks, tags)]

	@property
	def params(self):
		return self._params

	@params.setter
	def params(self, value):
		assert len(value) == len(self.datas)
		for v, T in zip(value, self.Ts):
			assert len(v) == 2
			q_mu, q_sigma_inv = v
			assert q_mu.shape == q_sigma_inv.shape == (T, D)

		self._params = value

 	def _initialize_variational_params(self, data, input, mask, tag):
 		T = data.shape[0]
		if isinstance(self.model, _LinearEmissions):
			q_mu = _initialize_linear_emissions_mean(model, data)

		else:
			q_mu = np.zeros((T, self.D))

		# Initialize the block tridiagonal precision matrix
		J_diag = np.tile(np.eye(self.D)[None, :, :], (T, 1, 1))
		J_offdiag = np.zeros((T-1, self.D, self.D))

	def sample(self):
		raise NotImplementedError

	def log_density(self, sample):
		assert isinstance(sample, list) and len(sample) == self.len(datas)

		logq = 0
		for s, (q_mu, q_sigma_inv) in zip(sample, self.params):
			raise NotImplementedError

		return logq

	