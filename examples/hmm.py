import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

from ssm.models import GaussianHMM, StudentsTHMM, AutoRegressiveHMM

# Set the parameters of the HMM
T = 500     # number of time bins
K = 5       # number of discrete states
D = 2       # number of observed dimensions

# Make an HMM with the true parameters
true_hmm = GaussianHMM(K, D)
z, y = true_hmm.sample(T)
z_test, y_test = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y)

# Fit models
N_sgd_iters = 1000
N_em_iters = 100

print("Fitting HMM with SGD")
hmm = GaussianHMM(K, D)
hmm_sgd_lls = hmm.fit(y, method="sgd", num_iters=N_sgd_iters)
hmm_sgd_test_ll = hmm.log_probability(y_test)
hmm_sgd_smooth = hmm.smooth(y)

print("Fitting HMM with EM")
hmm = GaussianHMM(K, D)
hmm_em_lls = hmm.fit(y, method="em", num_em_iters=N_em_iters)
hmm_em_test_ll = hmm.log_probability(y_test)
hmm_em_smooth = hmm.smooth(y)

print("Fitting Student's t HMM with SGD")
thmm = StudentsTHMM(K, D)
thmm_sgd_lls = thmm.fit(y, method="sgd", num_iters=N_sgd_iters)
thmm_sgd_test_ll = thmm.log_probability(y_test)
thmm_sgd_smooth = thmm.smooth(y)

print("Fitting ARHMM with SGD")
arhmm = AutoRegressiveHMM(K, D)
arhmm_sgd_lls = arhmm.fit(y, method="sgd", num_iters=N_sgd_iters)
arhmm_sgd_test_ll = arhmm.log_probability(y_test)
arhmm_sgd_smooth = arhmm.smooth(y)

print("Fitting ARHMM with EM")
arhmm = AutoRegressiveHMM(K, D)
arhmm_em_lls = arhmm.fit(y, method="em", num_em_iters=N_em_iters)
arhmm_em_test_ll = arhmm.log_probability(y_test)
arhmm_em_smooth = arhmm.smooth(y)

# Plot smoothed observations
plt.figure()
for d in range(D):
	plt.subplot(D, 1, d+1)
	plt.plot(y, '-k', lw=2, label="true")
	plt.plot(hmm_sgd_smooth, '-r', lw=1, label="HMM (SGD)")
	plt.plot(hmm_em_smooth, ':r', lw=1, label="HMM (EM)")
	plt.plot(thmm_sgd_smooth, ':g', lw=1, label="tHMM (EM)")
	plt.plot(arhmm_sgd_smooth, '-b', lw=1, label="ARHMM (SGD)")
	plt.plot(arhmm_em_smooth, ':b', lw=1, label="ARHMM (EM)")
	plt.legend(loc="upper right")

# Plot log likelihoods
plt.figure()
plt.plot(hmm_sgd_lls, label="HMM (SGD)")
plt.plot(hmm_em_lls, label="HMM (EM)")
plt.plot(thmm_sgd_lls, label="tHMM (SGD)")
plt.plot(arhmm_sgd_lls, label="ARHMM (SGD)")
plt.plot(arhmm_em_lls, label="ARHMM (EM)")
plt.plot(true_ll * np.ones(max(N_em_iters, N_sgd_iters)), ':', label="true")
plt.legend(loc="lower right")

print("Test log likelihood")
print("True: ", true_hmm.log_probability(y_test))
print("HMM (SGD) ", hmm_sgd_test_ll)
print("HMM (EM) ", hmm_em_test_ll)
print("tHMM (SGD) ", thmm_sgd_test_ll)
print("ARHMM (SGD) ", arhmm_sgd_test_ll)
print("ARHMM (EM) ", arhmm_em_test_ll)

plt.show()
