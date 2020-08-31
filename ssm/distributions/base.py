from warnings import warn
from functools import partial

import jax.numpy as np
import jax.scipy.special as spsp
import jax.scipy.stats as spst
from jax.flatten_util import ravel_pytree

from ssm.optimizers import minimize, convex_combination
from ssm.util import sum_tuples, weighted_sum_stats, \
    format_dataset, num_datapoints


class Distribution(object):
    """
    Base class for distributions with generic interface for maximum
    likelihood (or maximum a posteriori) estimation.
    """
    def __init__(self, prior=None):
        self.prior = prior

    @classmethod
    def from_example(cls, data, **kwargs):
        raise NotImplementedError

    @property
    def dimension(self):
        raise NotImplementedError

    @property
    def unconstrained_params(self):
        return ()

    @unconstrained_params.setter
    def unconstrained_params(self, value):
        pass

    def log_prior(self):
        """Evaluate prior probability of this distribution's parameters
        using its `self.prior` Distribution object.
        """
        raise NotImplementedError

    def log_prob(self, data, **kwargs):
        """Evaluate the log probability of a data array.
        """
        raise NotImplementedError

    def mode(self):
        """Return the mode of the distribution.
        """
        raise NotImplementedError

    def sample(self, rng, **kwargs):
        """Return a sample from the distribution.
        """
        raise NotImplementedError

    @format_dataset
    def fit_proximal(self, dataset, step_size,
                     optimizer_state=None,
                     callback=None):
        """Maximize the log joint but don't stray too far from
        the current parameters, as measured by the L2 distance.
        """
        # Convert step_size in [0, 1] to variance in [0, \infty)
        variance = step_size / (1 - step_size)

        if optimizer_state is not None:
            proximal_point = optimizer_state["proximal_point"]
            def regularizer(params):
                flat_params, _ = ravel_pytree(params)
                return 0.5 / variance * np.sum((flat_params - proximal_point)**2)
        else:
            regularizer = None

        scores = self.fit(dataset,
                          regularizer=regularizer,
                          callback=callback)

        optimizer_state = dict(
            proximal_point=ravel_pytree(self.unconstrained_params)[0],
            scores=scores)
        return optimizer_state

    @format_dataset
    def fit(self, dataset, regularizer=None, callback=None):
        """Fit the distribution parameters with maximum likelihood
        """
        def objective(params):
            # Set the params and evaluate log probability
            self.unconstrained_params = params
            lp = self.log_prior()
            for data_dict in dataset:
                _lp = self.log_prob(**data_dict)
                if "weights" in data_dict:
                    lp += (data_dict["weights"] * _lp).sum()
                else:
                    lp += _lp.sum()

            obj = -lp / num_datapoints(dataset)
            if regularizer is not None:
                obj += regularizer(params)
            return obj

        # Track the objective over iterations
        scores = []
        def _callback_wrapper(params):
            scores.append(objective(params))
            if callback is not None:
                callback(params)

        # minimize the objective via BFGS
        result = minimize(objective,
                          self.unconstrained_params,
                          callback=_callback_wrapper)
        if not result.success:
            warn("fit: minimize failed with result: {}".format(result))

        # Set the unconstrained parameters
        self.unconstrained_params = result["x"]
        return np.array(scores)


# Initialize a dictionary of conjugate prior mappings.
# This will be filled in by __init__.py.
CONJUGATE_PRIORS = dict()
def register_conjugate_prior(distribution, prior):
    assert issubclass(distribution, ExponentialFamilyDistribution) or \
           issubclass(distribution, CompoundDistribution)
    assert issubclass(prior, ConjugatePrior)
    CONJUGATE_PRIORS[distribution] = prior


class ExponentialFamilyDistribution(Distribution):
    """An interface for exponential family distributions
    with the necessary functionality for MAP estimation.
    """
    def __init__(self, prior=None):
        if prior is None:
            cls = CONJUGATE_PRIORS[self.__class__]
            prior = cls(dimension=self.dimension)
        self.prior = prior

    def sufficient_statistics(self, data, **kwargs):
        """
        Return the sufficient statistics for each datapoint in an array,
        This function should assume the leading dimensions give the batch
        size.
        """
        raise NotImplementedError

    def fit_expfam_with_stats(self, sufficient_statistics, num_datapoints):
        """Compute the maximum a posteriori (MAP) estimate of the distribution
        parameters, given the sufficient statistics of the data and the number
        of datapoints.
        """
        # Compute the posterior distribution given sufficient statistics
        posterior_stats = sum_tuples(self.prior.pseudo_obs, sufficient_statistics)
        posterior_counts = self.prior.pseudo_counts + num_datapoints
        posterior = self.prior.from_stats(posterior_stats, posterior_counts)

        # Set distribution parameters to posterior mode
        for param, val in posterior.mode.items():
            setattr(self, param, val)

    @format_dataset
    def fit(self, dataset):
        """Compute the maximum a posteriori (MAP) estimate of the distribution
        parameters.  For uninformative priors, this reduces to the maximum
        likelihood estimate.
        """
        # Compute the sufficient statistics and the number of datapoints
        suff_stats = None
        for data_dict in dataset:
            weights = data_dict["weights"] if "weights" in data_dict else None
            these_stats = weighted_sum_stats(
                self.sufficient_statistics(**data_dict),
                weights)
            suff_stats = sum_tuples(suff_stats, these_stats)

        return self.fit_expfam_with_stats(suff_stats, num_datapoints(dataset))

    @format_dataset
    def fit_proximal(self, dataset, step_size,
                     optimizer_state=None,
                     scale_factor=1.0,
                     callback=None):
        """Maximize the log joint but don't stray too far from
        the current parameters, as measured by the KL divergence.
        This boils down to taking a convex combination of sufficient
        statistics from this data and those that have been accumulated
        from past data.
        """
        # Compute the sufficient statistics and the number of datapoints
        suff_stats = None
        for data_dict in dataset:
            weights = data_dict["weights"] if "weights" in data_dict else None
            these_stats = weighted_sum_stats(
                self.sufficient_statistics(**data_dict), weights)
            suff_stats = sum_tuples(suff_stats, these_stats)

        # Scale the sufficient statistics by the given scale factor.
        # This is as if the sufficient statistics were accumulated
        # from the entire dataset rather than a batch.
        suff_stats = tuple(scale_factor * ss for ss in suff_stats)
        counts = scale_factor * num_datapoints(dataset)

        if optimizer_state is not None:
            # Take a convex combination of sufficient statistics from
            # this batch and those accumulated thus far.
            suff_stats = convex_combination(
                optimizer_state["suff_stats"], suff_stats, step_size)

            counts = convex_combination(optimizer_state["counts"], counts, step_size)

        self.fit_expfam_with_stats(suff_stats, counts)

        # Update the optimizer state
        optimizer_state = dict(suff_stats=suff_stats, counts=counts)
        return optimizer_state


class ConjugatePrior(Distribution):
    """Interface for a conjugate prior distribution.
    By default, the constructor should default to an uninformative
    prior distribution."""
    def __init__(self, prior=None, dimension=None, **kwargs):
        pass

    @classmethod
    def from_stats(cls, stats, counts):
        """
        Construct an instance of the prior distribution given
        sufficient statistics and counts.
        """
        raise NotImplementedError

    @property
    def pseudo_obs(self):
        """Return the pseudo observations under this prior.
        These should match up with the sufficient statistics of
        the conjugate distribution.
        """
        raise NotImplementedError

    @property
    def pseudo_counts(self):
        """Return the pseudo observations under this prior."""
        raise NotImplementedError


class CompoundDistribution(Distribution):
    """Interface for compound distributions like the Student's t
    distribution and the negative binomial distribution.
    """
    def __init__(self, prior=None):
        if prior is None:
            cls = CONJUGATE_PRIORS[self.__class__]
            prior = cls(dimension=self.dimension)
        self.prior = prior

    @property
    def unconstrained_nonconj_params(self):
        return ()

    @unconstrained_nonconj_params.setter
    def unconstrained_nonconj_params(self, value):
        pass

    def conditional_expectations(self, data, **kwargs):
        """Compute expectations under the conditional distribution
        over the auxiliary variables.  In the Student's t, for example,
        the auxiliary variables are the per-datapoint precision, :math:`\tau`,
        and the necessary expectations are :math:`\mathbb{E}[\tau]` and
        :math:`\mathbb{E}[\log \tau]`
        """
        raise NotImplementedError

    def expected_sufficient_statistics(self, expectations, data, **kwargs):
        """Compute expected sufficient statistics necessary for a conjugate
        update of some parameters.
        """
        raise NotImplementedError

    def expected_log_prob(self, expectations, data, **kwargs):
        """Compute the expected log probability.  This function will be
        optimized with respect to the remaining, non-conjugate parameters
        of the distribution.
        """
        raise NotImplementedError

    def e_step(self, dataset):
        # E step: compute conditional expectations
        return [self.conditional_expectations(**data_dict)
                for data_dict in dataset]

    def conjugate_m_step(self, dataset, expectations):
        # M step: find the optimal parameters for the conjugate part
        # of the compound distribution
        suff_stats = None
        for exps, data_dict in zip(expectations, dataset):
            weights = data_dict["weights"] if "weights" in data_dict else None
            these_stats = weighted_sum_stats(
                self.expected_sufficient_statistics(exps, **data_dict), weights)
            suff_stats = sum_tuples(suff_stats, these_stats)

        # Compute the posterior distribution given sufficient statistics
        posterior_stats = sum_tuples(self.prior.pseudo_obs, suff_stats)
        posterior_counts = self.prior.pseudo_counts + num_datapoints(dataset)
        posterior = self.prior.from_stats(posterior_stats, posterior_counts)

        # Set distribution parameters to posterior mode
        for param, val in posterior.mode.items():
            setattr(self, param, val)

    def nonconjugate_m_step(self, dataset, expectations):
        # M step: optimize the non-conjugate parameters via BFGS.
        def objective(params, itr):
            # Set the params
            self.unconstrained_nonconj_params = params
            lp = self.log_prior()
            for exps, data_dict in zip(expectations, dataset):
                weights = data_dict["weights"] if "weights" in data_dict else None
                _lp = self.expected_log_prob(exps, **data_dict)
                lp += np.sum(weights * _lp) if weights is not None else np.sum(_lp)
            return -lp / num_datapoints(dataset)

        # Minimize the objective via BFGS
        # NOTE: We use a weak tolerance so this doesn't take too long
        result = minimize(objective,
                          self.unconstrained_nonconj_params,
                          tol=1e-2)
        if not result.success:
            warn("fit: minimize failed with result: {}".format(result))

        # Set the unconstrained parameters
        self.unconstrained_nonconj_params = result['x']
        return result.fun

    @format_dataset
    def fit(self, dataset, num_iters=20, tol=1e-3):
        """Fit a compound distribution use EM.
        """
        log_probs = []
        converged = False
        while not converged:
            expectations = self.e_step(dataset)
            self.conjugate_m_step(dataset, expectations)
            log_probs.append(self.nonconjugate_m_step(dataset, expectations))

            if len(log_probs) > num_iters:
                # print("Max iters (", num_iters, ") reached")
                converged = True
            elif len(log_probs) >= 2 and abs(log_probs[-1] - log_probs[-2]) < tol:
                # print("log prob converged in ", len(log_probs), "iterations")
                converged = True
