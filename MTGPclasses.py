import torch
import gpytorch
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,n_tasks,kernel_type):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_tasks]))
        
        if kernel_type == 'RBF':         
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([n_tasks])),
                batch_shape=torch.Size([n_tasks])
            )
        if kernel_type == 'lin':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.LinearKernel()(batch_shape=torch.Size([n_tasks])),
                batch_shape=torch.Size([n_tasks])
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )