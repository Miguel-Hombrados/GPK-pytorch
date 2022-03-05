import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from torch import nn
import sys

from pathlib import Path
ProjectPath = Path.cwd()
utilsPath = Path.joinpath(ProjectPath,"utils")


UTIL_DIR = utilsPath
sys.path.append(
    str(UTIL_DIR)
)


from EvaluateConfidenceIntervals import EvaluateConfidenceIntervals
train_x = torch.linspace(0, 1, 400)

train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

    
        
test_x = torch.linspace(0, 1, 300)  
test_y = torch.stack([
    torch.sin(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * 0.2,
    torch.cos(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * 0.2,
], -1)
    

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)
model_val = BatchIndependentMultitaskGPModel(test_x, test_y, likelihood)
# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 100


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

# Set into eval mode
#model.eval()
#likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    
    f = model.forward(test_x)
    predictions = likelihood(f)
    mean = predictions.mean
    std = predictions.stddev
    lower, upper = predictions.confidence_region()
        

[IC1,IC2] = EvaluateConfidenceIntervals(test_y,mean,torch.pow(std,1))
mIC1 = torch.mean(IC1)
mIC2 = torch.mean(IC2)

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

# Plot training data as black stars

y1_ax.plot(test_x.detach().numpy(), test_y[:, 0].detach().numpy(), 'k+')
# Predictive mean as blue line
y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(test_x.numpy(), mean[:,0]-2*std[:,0], mean[:,0]+2*std[:,0], alpha=0.5)
y1_ax.set_ylim([-3, 3])
y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
#y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k+')
y2_ax.plot(test_x.detach().numpy(), test_y[:, 1].detach().numpy(), 'k+')
# Predictive mean as blue line
y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(test_x.numpy(), mean[:,1]-2*std[:,1], mean[:,1]+2*std[:,1], alpha=0.5)
y2_ax.set_ylim([-3, 3])
y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y2_ax.set_title('Observed Values (Likelihood)')





# factor = 0.5
# y1_ax.plot(test_x.detach().numpy(), test_y[:, 0].detach().numpy(), 'k+')
# # Predictive mean as blue line
# y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
# # Shade in confidence
# y1_ax.fill_between(test_x.numpy(), factor*lower[:, 0].numpy(), factor*upper[:, 0].numpy(), alpha=0.5)
# y1_ax.set_ylim([-3, 3])
# y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
# y1_ax.set_title('Observed Values (Likelihood)')

# # Plot training data as black stars
# #y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k+')
# y2_ax.plot(test_x.detach().numpy(), test_y[:, 1].detach().numpy(), 'k+')
# # Predictive mean as blue line
# y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
# # Shade in confidence
# y2_ax.fill_between(test_x.numpy(), factor*lower[:, 1].numpy(), factor*upper[:, 1].numpy(), alpha=0.5)
# y2_ax.set_ylim([-3, 3])
# y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
# y2_ax.set_title('Observed Values (Likelihood)')
# plt.draw()
None