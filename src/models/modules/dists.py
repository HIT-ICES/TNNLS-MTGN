import torch
import torch.distributions as D
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions.kl import register_kl
import torch.nn as nn
from torch import Tensor
import math
from numbers import Number,Real

import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints
from typing import Dict


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Examples::

        # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
        # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
                     torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        # Construct a batch of 3 Gaussian Mixture Models in 2D each
        # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._component_distribution, Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                 "compatible with `component_distribution."
                                 "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = \
            self._component_distribution.expand(batch_shape_comp)
        new._mixture_distribution = \
            self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(batch_shape=batch_shape,
                                               event_shape=event_shape,
                                               validate_args=False)
        new._validate_args = self._validate_args
        return new


    @constraints.dependent_property
    def support(self):
        # FIXME this may have the wrong shape when support contains batched
        # parameters
        return self._component_distribution.support

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs * self.component_distribution.variance,
                                  dim=-1 - self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1 - self._event_ndims)
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)


    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]


    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)


    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x

    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.mixture_distribution,
                                             self.component_distribution)
        return 'MixtureSameFamily' + '(' + args_string + ')'

class TruncatedNormal(ExponentialFamily):
    r"""
    Create a truncated normal distribution parameterized by :attr:`loc`, :attr:`scale`, :attr:`a` and :attr:`b`.
    
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution (often referred to as sigma)
        a (float or Tensor, Optional): low bound of the distribution. default: :obj:`-inf` 
        b (float or Tensor, Optional): high bound of the distribution. default: :obj:`inf`
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'a': constraints.real, 'b': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_mean = 0
    
    @property
    def alpha(self):
        return (self.a - self.loc)/self.scale
    
    @property
    def beta(self):
        return (self.b - self.loc)/self.scale
    
    @property
    def Z(self):
        if torch.any(torch.isinf(self.a)):
            return self.Phi(self.beta)
        elif torch.any(torch.isinf(self.b)):
            return -self.Phi(self.alpha)
        return self.Phi(self.beta) - self.Phi(self.alpha)

    @property
    def mean(self):
        if torch.any(torch.isinf(self.a)):
            return self.loc + self.scale * (- self.phi(self.beta))/self.Z
        elif torch.any(torch.isinf(self.b)):
            self.loc + self.scale * (self.phi(self.alpha))/self.Z
        return self.loc + self.scale * (self.phi(self.alpha) - self.phi(self.beta))/self.Z
    
    
    @property
    def variance(self):
        if torch.any(torch.isinf(self.a)):
            return self.scale.pow(2)*(1 - self.beta * (self.phi(self.beta) * self.Phi(self.beta).reciprocal()) - (self.phi(self.beta)/self.Phi(self.beta)).pow(2))
        elif torch.any(torch.isinf(self.b)):
            return self.scale.pow(2)*(1 + self.alpha * (self.phi(self.alpha) * self.Z.reciprocal()) - (self.phi(self.alpha)* self.Z.reciprocal()).pow(2))
        else:
            return self.scale.pow(2) * (1 + (self.alpha * self.phi(self.alpha) - self.beta * self.phi(self.beta)) * self.Z.reciprocal() - ((self.phi(self.alpha) - self.phi(self.beta))*self.Z.reciprocal()).pow(2))
    
    @property
    def entropy(self):
        if torch.any(torch.isinf(self.a)):
            return torch.log(math.sqrt(2*math.pi*math.e)*self.Z) - 0.5 * self.beta * self.phi(self.beta) * self.Z.reciprocal()
        elif torch.any(torch.isinf(self.b)):
            return torch.log(math.sqrt(2*math.pi*math.e)*self.Z) + 0.5 * self.alpha * self.phi(self.alpha) * self.Z.reciprocal()
        else:
            return torch.log(math.sqrt(2*math.pi*math.e)*self.Z) + 0.5 * (self.alpha * self.phi(self.alpha) - self.beta * self.phi(self.beta)) * self.Z.reciprocal()
    
    def __init__(self, loc, scale, a=float('-inf'), b=float('inf'), validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(TruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        super(TruncatedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    
    def iPhi(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return math.sqrt(2)*torch.erfinv(2*x - 1)
    
    def xi(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return (x - self.loc)/self.scale
    
    def Phi(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    def phi(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return torch.exp(-0.5* x.pow(2))/math.sqrt(2*math.pi)
    
    def cdf(self, x):
        if self._validate_args:
            self._validate_sample(x)
        xi = self.xi(x)
        return (self.Phi(xi) - self.Phi(self.alpha))/self.Z
    
    def pdf(self, x):
        if self._validate_args:
            self._validate_sample(x)
        xi = self.xi(x)
        return torch.where(torch.logical_and(x > self.a, x < self.b), self.phi(xi)/(self.Z * self.scale), torch.zeros_like(x))
    
    def icdf(self, x):
        if self._validate_args:
            self._validate_sample(x)
        return self.loc + self.scale * self.iPhi(self.Phi(self.alpha) + self.Z * x)
     
    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        xi = self.xi(x)
        prob = self.phi(xi)/(self.Z * self.scale)
        prob = torch.where(prob>0, prob, torch.ones_like(prob)*(1e-14)) 
        return torch.where(torch.logical_and(x > self.a, x < self.b), torch.log(prob), torch.ones_like(x)*(-14))

    # TODO: we may implement a rejection sampleer
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            #source = D.Uniform(self.Phi(self.alpha), self.Phi(self.beta)).sample(shape).to(self.Z.device)
            source = torch.rand(shape).to(self.Z.device) + 1e-6 # to avoid 0
            #source = source * self.Phi(self.beta)
            return self.icdf(source)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(eps)
        
    
    
class LogNormalMixtureDistribution(D.TransformedDistribution):
    """Mixture of log-normal distributions, which is modeled as follows:
    
    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_event_time * x + mean_log_inter_event_time
    z = exp(y)
    
    Args:
        locs: _description_
    """
    def __init__(
        self,
        locs: Tensor,
        log_scales: Tensor,
        log_weights: Tensor,
        mean_log_inter_event_time: float = 0.0,
        std_log_inter_event_time: float = 1.0,         
    ):
        mixture_dist = D.Categorical(probs=log_weights)
        component_dist = D.Normal(loc=locs, scale=log_scales.exp())
        gmm = D.MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_event_time == 0.0 and std_log_inter_event_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_event_time, scale=std_log_inter_event_time)]
        self.mean_log_inter_event_time = mean_log_inter_event_time
        self.std_log_inter_event_time = std_log_inter_event_time
        transforms.append(D.ExpTransform())
        super().__init__(gmm, transforms)
    
    @property
    def mean(self):
        """Compute the expected value of the distribution.
        """
        #a = self.std_log_inter_event_time
        #b = self.mean_log_inter_event_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.probs
        m = log_weights * torch.exp(loc + 0.5 * variance)
        return m.sum(-1)

class UpperTailTruncatedLogNormalMixtureDistribution(D.TransformedDistribution):
    """Mixture of truncated log-normal distributions, which is modeled as follows:
    
    x ~ TruncatedNormalMixtureModel(locs, log_scales, log_weights, lower_bound, log_upper_bound)
    y = std_log_inter_event_time * x + mean_log_inter_event_time
    z = exp(y)
    """
    def __init__(
        self,
        locs: Tensor,
        log_scales: Tensor,
        log_weights: Tensor,
        log_upper_bound: float = None,
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = UpperTailTruncatedNormal(loc=locs, scale=log_scales.exp(), b=log_upper_bound)
        gmm = D.MixtureSameFamily(mixture_dist, component_dist)
        transforms = [D.ExpTransform()]
        super().__init__(gmm, transforms)
    


class UpperTailTruncatedNormal(ExponentialFamily):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_mean = 0
    
        
    @property
    def beta(self):
        return (self.b - self.loc) / self.scale
    
    def phi(self, x):
        return torch.exp(-x*x/2)/math.sqrt(2 * math.pi)
    
    @property
    def mean(self):
        return self.loc - self.scale * (self.phi(self.beta)/self.Z)
    
    
    
    @property
    def Z(self):
        return self.cdf0(self.beta)
    
    @property
    def stddev(self):
        return self.scale*torch.sqrt(1 - self.beta*self.phi(self.beta)/self.Z - torch.power(self.phi(self.beta)/self.Z, 2.0))
    
    @property
    def variance(self):
        return self.stddev.pow(2)
    
    def __init__(self, loc, scale, b, validate_args=None):
        self.loc, self.scale, self.b = broadcast_all(loc, scale, b)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(UpperTailTruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
    
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            source = torch.rand(shape).to(self.Z.device)
            return self.icdf(source)
    
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(eps)
    
    def log_prob0(self, value):
        if self._validate_args:
            self._validate_sample(value)
            var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.where(value < self.b,
                            torch.log(self.prob(value) + 1e-14),
                            torch.ones_like(value)*(-14))

    def prob(self, value):
        """
        To allow differentiable optimization
        over zero probabilities.
        """
        if self._validate_args:
            self._validate_sample(value)
        return torch.where(value < self.b,
                        self.phi((value - self.loc) * self.scale.reciprocal())/(self.scale*self.Z), torch.zeros_like(value))
 
    def cdf0(self, value):
        """
        The untruncated CDF.
        """
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf(value/ math.sqrt(2)))
    
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = (value - self.loc) * self.scale.reciprocal()
        return (self.cdf0(value))/self.Z
    
    def icdf0(self, value):
        """
        The untruncated i-CDF.
        """
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.icdf0(self.Z*value)

    def entropy(self):
        return torch.log(math.sqrt(2*math.pi*math.e)*self.scale*self.Z) + -self.beta*self.phi(self.beta)/(2*self.Z)
        
@register_kl(LogNormalMixtureDistribution, LogNormalMixtureDistribution)
def kl_logNormMix_logNormMix(p: LogNormalMixtureDistribution, q: LogNormalMixtureDistribution):
    
    """Monto Carlo approximation of KL(p || q).
    KL(p || q) = E_p[log(p(x)) - log(q(x))] \approx \frac{1}{N} \sum_{i=1}^N log(p(x_i)/q(x_i))
    """
    NUM_of_MC_SAMPLES = 5000   # number of samples for Monte Carlo approximation
    samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    px = p.log_prob(samples)
    qx = q.log_prob(samples)
    return torch.mean(px - qx)


@register_kl(UpperTailTruncatedLogNormalMixtureDistribution, LogNormalMixtureDistribution)
def kl_TruncLogNormMix_logNormMix(p: UpperTailTruncatedLogNormalMixtureDistribution, q: LogNormalMixtureDistribution):
    NUM_of_MC_SAMPLES = 10   # number of samples for Monte Carlo approximation
    # FIXME: Here i am not sure why some times the samples will be negative.. just ignore them (very rare)
    samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    px = p.log_prob(samples)
    qx = q.log_prob(samples)
    return torch.mean(px - qx, dim=-1)

@register_kl(UpperTailTruncatedLogNormalMixtureDistribution, D.MixtureSameFamily)
def kl_TruncLogNormMix_MixtureSameFamily(p: UpperTailTruncatedLogNormalMixtureDistribution, q: D.MixtureSameFamily):
    NUM_of_MC_SAMPLES = 10
    samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    while not torch.all(samples > 0):
        samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    px = p.log_prob(samples)
    qx = q.log_prob(samples)
    return torch.mean(px - qx, dim=-1)

@register_kl(D.MixtureSameFamily, D.MixtureSameFamily)
def kl_TruncLogNormMix_MixtureSameFamily(p: D.MixtureSameFamily, q: D.MixtureSameFamily):
    NUM_of_MC_SAMPLES = 10
    samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    while not torch.all(samples > 0):
        samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    px = p.log_prob(samples)
    qx = q.log_prob(samples)
    return torch.mean(px - qx, dim=-1)

@register_kl(D.MixtureSameFamily, LogNormalMixtureDistribution)
def kl_TruncLogNormMix_MixtureSameFamily(p: D.MixtureSameFamily, q: D.MixtureSameFamily):
    NUM_of_MC_SAMPLES = 15
    samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    #samples = samples.clamp(min=1e-6)
    px = p.log_prob(samples)
    qx = q.log_prob(samples)
    return torch.mean(px - qx, dim=-1)

@register_kl(MixtureSameFamily, LogNormalMixtureDistribution)
def kl_TruncLogNormMix_MixtureSameFamily(p: D.MixtureSameFamily, q: D.MixtureSameFamily):
    NUM_of_MC_SAMPLES = 10
    samples = p.sample(sample_shape=(NUM_of_MC_SAMPLES,))
    samples = samples.clamp(min=1e-6)
    #samples = samples.clamp(min=1e-6)
    px = p.log_prob(samples)
    qx = q.log_prob(samples)
    return torch.mean(px - qx, dim=-1)