# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:01:21 2022

@author: mahom
"""
# import positivity constraint
import torch
import gpytorch
from gpytorch.constraints import Positive

class bias(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, bias_prior=None, bias_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(
            name='raw_bias', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set the parameter constraint to be positive, when nothing is specified
        if bias_constraint is None:
            bias_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_bias", bias_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if bias_prior is not None:
            self.register_prior(
                "bias_prior",
                bias_prior,
                lambda m: m.bias,
                lambda m, v : m._set_bias(v),
            )

    # now set up the 'actual' paramter
    @property
    def bias(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_bias_constraint.transform(self.raw_bias)

    @bias.setter
    def bias(self, value):
        return self._set_bias(value)

    def _set_bias(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bias)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_bias=self.raw_bias_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply biasscale
        x1_ = x1
        x2_ = x2
        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, **params)
        diff0 = torch.ones_like(diff)*self.bias
        # prevent divide by 0 errors
        diff0.where(diff0 == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return diff0