import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

from ssm.models import HMM

# Set the parameters of the HMM
T = 500     # number of time bins
K = 5       # number of discrete states
D = 2       # number of observed dimensions

# Make an HMM with the true parameters
true_hmm = HMM(K, D, observations="diagonal_gaussian")
z, y = true_hmm.sample(T)
z_test, y_test = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y)

# Fit models
N_sgd_iters = 1000
N_em_iters = 100

print("Fitting diagonal Gaussian HMM with SGD")
hmm = HMM(K, D, observations="diagonal_gaussian")
hmm_sgd_lls = hmm.fit(y, method="sgd", num_iters=N_sgd_iters)
hmm_sgd_test_ll = hmm.log_probability(y_test)
hmm_sgd_smooth = hmm.smooth(y)

print("Fitting diagonal Gaussian HMM with EM")
hmm = HMM(K, D, observations="diagonal_gaussian")
hmm_em_lls = hmm.fit(y, method="em", num_em_iters=N_em_iters)
hmm_em_test_ll = hmm.log_probability(y_test)
hmm_em_smooth = hmm.smooth(y)

print("Fitting full Gaussian HMM with SGD")
mvnhmm = HMM(K, D, observations="gaussian")
mvnhmm_sgd_lls = hmm.fit(y, method="sgd", num_iters=N_sgd_iters)
mvnhmm_sgd_test_ll = hmm.log_probability(y_test)
mvnhmm_sgd_smooth = hmm.smooth(y)

print("Fitting full Gaussian HMM with EM")
mvnhmm = HMM(K, D, observations="gaussian")
mvnhmm_em_lls = hmm.fit(y, method="em", num_em_iters=N_em_iters)
mvnhmm_em_test_ll = hmm.log_probability(y_test)
mvnhmm_em_smooth = hmm.smooth(y)

print("Fitting Student's t HMM with SGD")
thmm = HMM(K, D, observations="studentst")
thmm_sgd_lls = thmm.fit(y, method="sgd", num_iters=N_sgd_iters)
thmm_sgd_test_ll = thmm.log_probability(y_test)
thmm_sgd_smooth = thmm.smooth(y)

print("Fitting Student's t HMM with EM")
thmm = HMM(K, D, observations="studentst")
thmm_em_lls = thmm.fit(y, method="em", num_em_iters=N_em_iters)
thmm_em_test_ll = thmm.log_probability(y_test)
thmm_em_smooth = thmm.smooth(y)

print("Fitting ARHMM with SGD")
arhmm = HMM(K, D, observations="ar")
arhmm_sgd_lls = arhmm.fit(y, method="sgd", num_iters=N_sgd_iters)
arhmm_sgd_test_ll = arhmm.log_probability(y_test)
arhmm_sgd_smooth = arhmm.smooth(y)

print("Fitting ARHMM with EM")
arhmm = HMM(K, D, observations="ar")
arhmm_em_lls = arhmm.fit(y, method="em", num_em_iters=N_em_iters)
arhmm_em_test_ll = arhmm.log_probability(y_test)
arhmm_em_smooth = arhmm.smooth(y)

print("Fitting tARHMM with SGD")
tarhmm = HMM(K, D, observations="robust_ar")
tarhmm_sgd_lls = tarhmm.fit(y, method="sgd", num_iters=N_sgd_iters)
tarhmm_sgd_test_ll = tarhmm.log_probability(y_test)
tarhmm_sgd_smooth = tarhmm.smooth(y)

print("Fitting tARHMM with EM")
tarhmm = HMM(K, D, observations="robust_ar")
tarhmm_em_lls = tarhmm.fit(y, method="em", num_em_iters=N_em_iters)
tarhmm_em_test_ll = tarhmm.log_probability(y_test)
tarhmm_em_smooth = tarhmm.smooth(y)

# Plot smoothed observations
plt.figure()
for d in range(D):
	plt.subplot(D, 1, d+1)
	plt.plot(y, '-k', lw=2, label="true")
	l1 = plt.plot(hmm_sgd_smooth, '-', lw=1,
                      label="HMM (SGD)")[0]
	plt.plot(hmm_em_smooth, ':', lw=1, color=l1.get_color(),
                 label="HMM (EM)")
	l2 = plt.plot(thmm_sgd_smooth, '-', lw=1,
                      label="tHMM (SGD)")[0]
	plt.plot(thmm_em_smooth, ':', lw=1, color=l2.get_color(),
                 label="tHMM (EM)")
	l3 = plt.plot(arhmm_sgd_smooth, '-', lw=1,
                      label="ARHMM (SGD)")[0]
	plt.plot(arhmm_em_smooth, ':', lw=1, color=l3.get_color(),
                 label="ARHMM (EM)")
	l4 = plt.plot(tarhmm_sgd_smooth, '-', lw=1,
                      label="tARHMM (SGD)")[0]
	plt.plot(arhmm_em_smooth, ':', lw=1, color=l4.get_color(),
                 label="tARHMM (EM)")
	plt.legend(loc="upper right")

# Plot log likelihoods
plt.figure()
l1 = plt.plot(hmm_sgd_lls, ls='-', label="HMM (SGD)")[0]
plt.plot(hmm_em_lls, ls=':', label="HMM (EM)", color=l1.get_color())
l1b = plt.plot(mvnhmm_sgd_lls, ls='-', label="MVN HMM (SGD)")[0]
plt.plot(mvnhmm_em_lls, ls=':', label="MVN HMM (EM)", color=l1b.get_color())
l2 = plt.plot(thmm_sgd_lls, ls='-', label="tHMM (SGD)")[0]
plt.plot(thmm_em_lls, ls=':', label="tHMM (EM)", color=l2.get_color())
l3 = plt.plot(arhmm_sgd_lls, ls='-', label="ARHMM (SGD)")[0]
plt.plot(arhmm_em_lls, ls=':', label="ARHMM (EM)", color=l3.get_color())
l4 = plt.plot(tarhmm_sgd_lls, ls='-', label="tARHMM (SGD)")[0]
plt.plot(tarhmm_em_lls, ls=':', label="tARHMM (EM)", color=l4.get_color())
plt.plot(true_ll * np.ones(max(N_em_iters, N_sgd_iters)), ':', label="true")
plt.legend(loc="lower right")

print("Test log likelihood")
print("True: ", true_hmm.log_probability(y_test))
print("HMM (SGD) ", hmm_sgd_test_ll)
print("HMM (EM) ", hmm_em_test_ll)
print("MVN HMM (SGD) ", mvnhmm_sgd_test_ll)
print("MVN HMM (EM) ", mvnhmm_em_test_ll)
print("tHMM (SGD) ", thmm_sgd_test_ll)
print("tHMM (EM) ", thmm_em_test_ll)
print("ARHMM (SGD) ", arhmm_sgd_test_ll)
print("ARHMM (EM) ", arhmm_em_test_ll)
print("tARHMM (SGD) ", tarhmm_sgd_test_ll)
print("tARHMM (EM) ", tarhmm_em_test_ll)

plt.show()
