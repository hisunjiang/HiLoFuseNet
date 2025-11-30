import torch
import torch.nn as nn
from .losses import soft_dtw
from .losses import path_soft_dtw

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss

class MSECorrSmoothLoss(nn.Module):
    def __init__(self, num_losses=3, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()
        # 每个任务的 log(sigma^2) 参数
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, pred, target):
        # MSE loss
        loss_mse = self.mse(pred, target)

        # Correlation loss
        vx = pred - pred.mean(dim=0, keepdim=True)
        vy = target - target.mean(dim=0, keepdim=True)
        corr_num = (vx * vy).sum(dim=0)
        corr_den = torch.sqrt((vx ** 2).sum(dim=0) * (vy ** 2).sum(dim=0)) + self.eps
        corr = torch.clamp(corr_num / corr_den, -1.0, 1.0)
        loss_corr = 1 - corr.mean()

        # Smoothness loss (二阶差分)
        loss_smooth = (pred[2:] - 2 * pred[1:-1] + pred[:-2]).pow(2).mean()

        # 三个loss合并
        losses = torch.stack([loss_mse, loss_corr, loss_smooth])

        # 加权总loss： precision * loss + log_var
        precision = torch.exp(-self.log_vars.to(pred.device))
        total_loss = (precision * losses + self.log_vars.to(pred.device)).sum()

        return total_loss

class MSEStructuralLoss(nn.Module):
    def __init__(self, num_losses=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='none')
        # 每个任务的 log(sigma^2) 参数
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, pred, target):
        # MSE loss
        mse_loss = self.mse(pred, target)

        # mean, var, cov
        target_mean = torch.mean(target, dim=0, keepdim=True)
        pred_mean = torch.mean(pred, dim=0, keepdim=True)

        target_var = torch.var(target, dim=0, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=0, keepdim=True, unbiased=False)
        target_std = torch.sqrt(target_var + 1e-5)
        pred_std = torch.sqrt(pred_var + 1e-5)

        target_pred_cov = torch.mean((target - target_mean) * (pred - pred_mean), dim=0, keepdim=True)

        # losses
        linear_corr = target_pred_cov / (target_std * pred_std)
        corr_loss = (1.0 - linear_corr).mean()

        target_softmax = torch.softmax(target, dim=0)
        pred_softmax = torch.log_softmax(pred, dim=0)
        var_loss = self.kl_loss(pred_softmax, target_softmax).sum(dim=0).mean()
        loss_smooth = (pred[2:] - 2 * pred[1:-1] + pred[:-2]).pow(2).mean()

        mean_loss = torch.abs(target_mean - pred_mean).mean()

        # 三个loss合并
        losses = torch.stack([corr_loss, loss_smooth])

        # 加权总loss： precision * loss + log_var
        precision = torch.exp(-self.log_vars.to(pred.device))
        total_loss = mse_loss + 0.4 *(precision * losses + self.log_vars.to(pred.device)).sum() #

        return total_loss

class DILATELoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        N_output = pred.shape[0] # trajectory length

        softdtw_batch = soft_dtw.SoftDTWBatch.apply
        D = torch.zeros((1, N_output, N_output)).to(pred.device)
        for k in range(1):
            Dk = soft_dtw.pairwise_distances(target.view(-1, 1), pred.view(-1, 1))
            D[k:k + 1, :, :] = Dk
        loss_shape = softdtw_batch(D, self.gamma)

        path_dtw = path_soft_dtw.PathDTWBatch.apply
        path = path_dtw(D, self.gamma)
        Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(pred.device)
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = loss_shape # self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss

class Structural_loss(nn.Module):
    def __init__(self, model, layer_name="proj"):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.model = model
        self.layer_name = layer_name
        self.mse = nn.MSELoss()
        self.ld = 0.5

    def forward(self, x, target):
        if self.model.training:
            pred = self.model(x)
        else:
            pred = x

        # MSE loss
        mse_loss = self.mse(pred, target)

        # mean, var, cov
        target_mean = torch.mean(target, dim=0, keepdim=True)
        pred_mean = torch.mean(pred, dim=0, keepdim=True)

        target_var = torch.var(target, dim=0, keepdim=True, unbiased=False)
        pred_var   = torch.var(pred, dim=0, keepdim=True, unbiased=False)
        target_std = torch.sqrt(target_var + 1e-5)
        pred_std   = torch.sqrt(pred_var + 1e-5)

        target_pred_cov = torch.mean((target - target_mean) * (pred - pred_mean), dim=0, keepdim=True)

        # losses
        linear_corr = target_pred_cov / (target_std * pred_std)
        corr_loss = (1.0 - linear_corr).mean()

        target_softmax = torch.softmax(target, dim=0)
        pred_softmax   = torch.log_softmax(pred, dim=0)
        var_loss = self.kl_loss(pred_softmax, target_softmax).sum(dim=0).mean()

        mean_loss = torch.abs(target_mean - pred_mean).mean()

        # ----------- 训练模式下：算梯度动态权重 ----------
        if self.model.training:
            params = list(getattr(self.model, self.layer_name).parameters())

            def grad_norm(loss):
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
                grads = [g for g in grads if g is not None]
                if len(grads) == 0:
                    return torch.tensor(0.0, device=loss.device, requires_grad=True)
                return torch.cat([g.reshape(-1) for g in grads]).norm()

            corr_grad_norm = grad_norm(corr_loss)
            var_grad_norm  = grad_norm(var_loss)
            mean_grad_norm = grad_norm(mean_loss)

            grad_avg = (corr_grad_norm + var_grad_norm + mean_grad_norm) / 3.0

            alpha = grad_avg.detach() / (corr_grad_norm.detach() + 1e-8)
            beta  = grad_avg.detach() / (var_grad_norm.detach() + 1e-8)
            gamma = grad_avg.detach() / (mean_grad_norm.detach() + 1e-8)

            linear_sim = (1.0 + linear_corr) * 0.5
            var_sim = (2 * target_std * pred_std) / (target_var + pred_var + 1e-5)
            gamma = gamma * torch.mean(linear_sim * var_sim).detach()

            return mse_loss + self.ld*(alpha * corr_loss + beta * var_loss + gamma * mean_loss)
        else:
            # ----------- 验证/测试模式下：直接算普通 loss ----------
            return mse_loss + self.ld*(corr_loss + var_loss + mean_loss) / 3.0



