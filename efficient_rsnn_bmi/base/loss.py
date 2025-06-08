import torch
import torch.nn as nn

from stork.loss_stacks import LossStack

class CSTLossStack(LossStack):
    def __init__(self, 
                 mask=None, 
                 density_weighting_func=False):
        super().__init__()
        self.mask = mask
        self.loss_fn = None  # to be defined in the child class
        self.density_weighting_func = density_weighting_func

    def get_R2(self, pred, target):
        # Julian Rossbroich
        # modified july 2024
        """
        Args:
            pred: Predicted series of the model (batch_size * timestep * nb_outputs),
            target: Ground truth series (batch_size * timestep * nb_outputs).

        Return:
            r2: R-squared between the inputs along consecutive axis, over a batch.
        """

        # For each feature, calculate R2
        # We use the mean across all samples to calculate sst
        # print(f"Target: {target}")
        # print(f"Target Shape: {target.shape}")
        # print(f"Prediction: {pred}")
        # print(f"Prediction Shape: {pred.shape}")
        ssr = torch.sum((target - pred) ** 2, dim=(0, 1))
        sst = torch.sum((target - torch.mean(target, dim=(0, 1))) ** 2, dim=(0, 1))
        # print(f"SSR: {ssr}")
        # print(f"SSR Shape: {ssr.shape}")
        # print(f"SST: {sst}")
        # print(f"SST: {sst}")
        r2 = (1 - ssr / sst).detach().cpu().numpy()
        # print(f"r2: {r2}")
        # print(f"r2 Shape: {r2.shape}")

        return [float(r2[0].round(3)), float(r2[1].round(3)), float(r2.mean().round(3))]

    def get_metric_names(self):
        # Julian Rossbroich
        # modified july 2024
        return ["r2x", "r2y", "r2"]

    def compute_loss(self, output, target):
        """Computes MSQE loss between output and target."""

        if self.mask is not None:
            output = output * self.mask.expand_as(output)
            target = target * self.mask.expand_as(output)
            
        if self.density_weighting_func:
            weight = self.density_weighting_func(target)
        else:
            weight = None

        self.metrics = self.get_R2(output, target)
        result = self.loss_fn(output, target, weight=weight)
        # print(f"Result: {result}")
        return result

    def predict(self, output):
        return output

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class MeanSquareError(CSTLossStack):
    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_MSEloss
        
    def _weighted_MSEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.mean(weight * (output - target) ** 2)
        else:
            return torch.mean((output - target) ** 2)


class RootMeanSquareError(CSTLossStack):

    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_RMSEloss

    def _weighted_RMSEloss(self, output, target, weight=None):
        # print(f"Output: {output}")
        # print(f"Output Shape: {output.shape}")
        # print(f"Target: {target}")
        # print(f"Target Shape: {target.shape}")
        if weight is not None:
            return torch.sqrt(torch.mean(weight * (output - target) ** 2))
        else:
            return torch.sqrt(torch.mean((output - target) ** 2))


class MeanAbsoluteError(CSTLossStack):
    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_MAEloss

    def _weighted_MAEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.mean(weight * torch.abs(output - target))
        else:
            return torch.mean(torch.abs(output - target))


class HuberLoss(CSTLossStack):
    def __init__(self, delta=1.0, mask=None, density_weighting_func=False):
        
        if density_weighting_func:
            raise ValueError("Density weighting not supported for Huber loss.")
        
        super().__init__(mask=mask)
        self.loss_fn = nn.SmoothL1Loss(beta=delta)
        self.delta = delta

class TeTLoss(CSTLossStack):
    def __init__(self, means=1.0, lamb=1e-3, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.means = means
        self.lamb = lamb
        self.mse = nn.MSELoss()
        self.loss_fn = self._weighted_TeTLoss

    def _rmse(self, output, target, weight=None):
        if weight is not None:
            return torch.sqrt(torch.mean(weight * (output - target) ** 2))
        else:
            return torch.sqrt(torch.mean((output - target) ** 2))
    
    def _weighted_TeTLoss(self, output, target, weight=None):
        T = output.size(1) # time step
        loss_tet = 0.0

        # print(f"Output: {output}")
        # print(f"Output Shape: {output.shape}")
        # print(f"Target: {target}")
        # print(f"Target Shape: {target.shape}")
        # print(f"Weight: {weight}")
        # print(f"Weight shape: {weight.shape}")
        for t in range(T):
            if weight: 
                loss_tet += self._rmse(output[:, t], target[:, t], weight[:, t])
            else:
                loss_tet += self._rmse(output[:, t], target[:, t])  
        loss_tet = loss_tet / T

        if self.lamb > 0:
            reg_target = torch.full_like(output, fill_value=self.means)
            loss_reg = self.mse(output, reg_target)
        else:
            loss_reg = 0.0

        total_loss = (1 - self.lamb) * loss_tet + self.lamb * loss_reg
        # self.metrics = [float(loss_tet.item()), float(loss_reg.item()), float(total_loss.item())]
        return total_loss
