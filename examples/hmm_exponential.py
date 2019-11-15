import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import ssm
from ssm.util import find_permutation

# Set the parameters of the HMM
T = 500     # number of time bins
K = 5       # number of discrete states
D = 2       # number of observed dimensions

# Make an HMM with the true parameters
true_hmm = ssm.HMM(K, D, observations="exponential")
z, y = true_hmm.sample(T)
z_test, y_test = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y)

# Fit models
N_sgd_iters = 1000
N_em_iters = 100

# A bunch of observation models that all include the
# diagonal Gaussian as a special case.
observations = [
    "exponential"
]

# Fit with both SGD and EM
methods = ["sgd", "em"]

results = {}
for obs in observations:
    for method in methods:
        print("Fitting {} HMM with {}".format(obs, method))
        model = ssm.HMM(K, D, observations=obs)
        train_lls = model.fit(y, method=method)
        test_ll = model.log_likelihood(y_test)
        smoothed_y = model.smooth(y)

        # Permute to match the true states
        model.permute(find_permutation(z, model.most_likely_states(y)))
        smoothed_z = model.most_likely_states(y)
        results[(obs, method)] = (model, train_lls, test_ll, smoothed_z, smoothed_y)

# Plot the inferred states
fig, axs = plt.subplots(len(observations) + 1, 1, figsize=(12, 8))

# Plot the true states
plt.sca(axs[0])
plt.imshow(z[None, :], aspect="auto", cmap="jet")
plt.title("true")
plt.xticks()

# Plot the inferred states
for i, obs in enumerate(observations):
    zs = []
    for method, ls in zip(methods, ['-', ':']):
        _, _, _, smoothed_z, _ = results[(obs, method)]
        zs.append(smoothed_z)

    plt.sca(axs[i+1])
    plt.imshow(np.row_stack(zs), aspect="auto", cmap="jet")
    plt.yticks([0, 1], methods)
    if i != len(observations) - 1:
        plt.xticks()
    else:
        plt.xlabel("time")
    plt.title(obs)

plt.tight_layout()

# Plot smoothed observations
fig, axs = plt.subplots(D, 1, figsize=(12, 8))

# Plot the true data
for d in range(D):
    if D==1:
        plt.sca(axs)
    else:
        plt.sca(axs[d])
    plt.plot(y[:, d], '-k', lw=2, label="True")
    plt.xlabel("time")
    plt.ylabel("$y_{{}}$".format(d+1))

for obs in observations:
    line = None
    for method, ls in zip(methods, ['-', ':']):
        _, _, _, _, smoothed_y = results[(obs, method)]
        for d in range(D):
            if D==1:
                plt.sca(axs)
            else:
                plt.sca(axs[d])
            color = line.get_color() if line is not None else None
            line = plt.plot(smoothed_y[:, d], ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]

# Make a legend
if D==1:
    plt.sca(axs)
else:
    plt.sca(axs[0])
plt.legend(loc="upper right")
plt.tight_layout()

# Plot log likelihoods
plt.figure(figsize=(12, 8))
for obs in observations:
    line = None
    for method, ls in zip(methods, ['-', ':']):
        _, lls, _, _, _ = results[(obs, method)]
        color = line.get_color() if line is not None else None
        line = plt.plot(lls, ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]

xlim = plt.xlim()
plt.plot(xlim, true_ll * np.ones(2), '-k', label="true")
plt.xlim(xlim)

plt.legend(loc="lower right")
plt.tight_layout()

# Print the test log likelihoods
print("Test log likelihood")
print("True: ", true_hmm.log_likelihood(y_test))
for obs in observations:
    for method in methods:
        _, _, test_ll, _, _ = results[(obs, method)]
        print("{} ({}): {}".format(obs, method, test_ll))

plt.show()

