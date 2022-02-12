import torch
import gpytorch
from mykernels import bias
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_task,kernel_type):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_task]))
        
        if  kernel_type == 'rbf':
            kernel_cov =  gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_task]))
        if  kernel_type == 'linear':
            kernel_cov =  gpytorch.kernels.LinearKernel(batch_shape=torch.Size([num_task])) 
        if  kernel_type == 'matern':  
            kernel_cov =  gpytorch.kernels.MaternKernel(nu  = 2.5, batch_shape=torch.Size([num_task]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_cov,
            batch_shape=torch.Size([num_task])
        ) + bias(batch_shape=torch.Size([num_task])) 


###  MultitaskMultivariateNorm
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x,validate_args = True)
        )  

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_task,kernel_type,option_lv):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_task
        )
        
        if  kernel_type == 'rbf':
            kernel_cov =  gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_task]))
        if  kernel_type == 'linear':
            kernel_cov =  gpytorch.kernels.LinearKernel(batch_shape=torch.Size([num_task]))
        if  kernel_type == 'matern':  
            kernel_cov =  gpytorch.kernels.MaternKernel(nu  = 2.5, batch_shape=torch.Size([num_task]))
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel_cov, num_tasks=num_task, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
