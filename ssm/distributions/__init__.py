from ssm.distributions.base import Distribution, \
    ExponentialFamilyDistribution, CompoundDistribution, ConjugatePrior, \
        register_conjugate_prior

from ssm.distributions.distributions import \
    Bernoulli, Beta, Binomial, Categorical, Dirichlet, Gamma, \
        MatrixNormalInverseWishart, MultivariateNormal, MultivariateStudentsT, \
            Normal, NormalInverseGamma, NormalInverseWishart, Poisson, StudentsT


from ssm.distributions.regressions import \
    LinearRegression, LinearAutoRegression, \
    MultivariateStudentsTLinearRegression, MultivariateStudentsTAutoRegression, \
    GeneralizedLinearModel

# Register conjugate priors
register_conjugate_prior(Bernoulli, Beta)
register_conjugate_prior(Binomial, Beta)
register_conjugate_prior(Categorical, Dirichlet)
register_conjugate_prior(MultivariateNormal, NormalInverseWishart)
register_conjugate_prior(LinearRegression, MatrixNormalInverseWishart)
register_conjugate_prior(LinearAutoRegression, MatrixNormalInverseWishart)
register_conjugate_prior(MultivariateStudentsT, NormalInverseWishart)
register_conjugate_prior(MultivariateStudentsTLinearRegression, MatrixNormalInverseWishart)
register_conjugate_prior(MultivariateStudentsTAutoRegression, MatrixNormalInverseWishart)
register_conjugate_prior(Normal, NormalInverseGamma)
register_conjugate_prior(Poisson, Gamma)
register_conjugate_prior(StudentsT, NormalInverseGamma)
