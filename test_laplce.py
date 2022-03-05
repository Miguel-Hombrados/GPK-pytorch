# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:18:53 2022

@author: mahom
"""
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
test_x = torch.linspace(0, 1, 51)
test_y = torch.sin(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * math.sqrt(0.04)

class ApproxGPModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(ApproxGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.LaplaceLikelihood()
model = ApproxGPModel(train_x)
import os
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 500

# Find optimal model hyperparameters
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the number of training datapoints
mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

for i in range(training_iterations):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
  
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    y_preds = likelihood(model(test_x))   
      
    f_preds = model(test_x)
    
    f_preds2 = model.forward(test_x)
    y_preds_t = likelihood(f_preds2.mean)
    gpytorch.settings.num_likelihood_samples._set_value(10)
    y_preds = likelihood(f_preds)
    y_preds1 = y_preds.mean
    y_preds2 = likelihood(f_preds.mean)
    observed_pred = likelihood(model(test_x))
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # Get upper and lower confidence bounds
    # Plot training data as black stars
    ax.plot(test_x.numpy(), test_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x, observed_pred.mean.detach().numpy()[9,:], 'bo')
    ax.plot(test_x, f_preds.mean.detach().numpy()[9,:], 'g+')
    plt.show()
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])