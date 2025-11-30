import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss

class MSESCLoss(nn.Module):
    def __init__(self, num_losses=2, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, pred, target):
        # MSE
        mse_loss = self.mse(pred, target)

        # mean, var, cov
        target_mean = torch.mean(target, dim=0, keepdim=True)
        pred_mean = torch.mean(pred, dim=0, keepdim=True)

        target_var = torch.var(target, dim=0, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=0, keepdim=True, unbiased=False)
        target_std = torch.sqrt(target_var + self.eps)
        pred_std = torch.sqrt(pred_var + self.eps)

        target_pred_cov = torch.mean((target - target_mean) * (pred - pred_mean), dim=0, keepdim=True)

        # Correlation
        linear_corr = target_pred_cov / (target_std * pred_std)
        corr_loss = (1.0 - linear_corr).mean()

        # Second-order difference smoothness
        smooth_loss = (pred[2:] - 2 * pred[1:-1] + pred[:-2]).pow(2).mean()

        losses = torch.stack([corr_loss, smooth_loss])

        # total loss: mse + weight * (precision * loss + log_var)
        precision = torch.exp(-self.log_vars.to(pred.device))
        total_loss = mse_loss + 0.4 *(precision * losses + self.log_vars.to(pred.device)).sum()

        return total_loss
