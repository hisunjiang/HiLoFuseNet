import torch
from scipy.stats import pearsonr

def train(loader, model, optimizer, loss_function, device):
    """
    Train the model for one epoch and compute training loss and Pearson correlation.

    Args:
        loader: DataLoader for training set
        model: PyTorch model
        optimizer: optimizer, e.g., torch.optim.Adam
        loss_function: loss function, e.g., nn.MSELoss()
        device: 'cuda' or 'cpu'

    Returns:
        avg_loss: float, average training loss for the epoch
        epoch_corr: list of float, Pearson correlation per output variable
    """
    model.train()  # set model to training mode
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        optimizer.zero_grad()  # reset gradients

        # forward pass
        output = model(batch_x)

        # compute loss
        loss = loss_function(output, batch_y)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # accumulate predictions and labels for performance metrics
        all_preds.append(output.detach().cpu())
        all_labels.append(batch_y.detach().cpu())

    # concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # compute Pearson correlation
    epoch_corr = []
    if all_preds.dim() == 2 and all_preds.shape[1] > 1:
        # multi-output regression
        num_vars = all_preds.shape[1]
        for i in range(num_vars):
            corr, _ = pearsonr(all_preds[:, i], all_labels[:, i])
            epoch_corr.append(corr)
    else:
        # single-output regression
        corr, _ = pearsonr(all_preds.squeeze(), all_labels.squeeze())
        epoch_corr.append(corr)

    avg_loss = total_loss / len(loader)

    return avg_loss, epoch_corr


def validation(loader, model, loss_function, device):
    """
    Compute validation loss and Pearson correlation for regression task.

    Args:
        loader: DataLoader for validation set
        model: PyTorch model
        loss_function: loss function, e.g., nn.MSELoss()
        device: 'cuda' or 'cpu'

    Returns:
        avg_loss: float, average validation loss
        epoch_corr: list of float, Pearson correlation per output variable
    """
    model.eval()  # set model to evaluation mode
    all_preds, all_labels = [], []

    with torch.no_grad():  # no gradient computation
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            output = model(batch_x)

            # accumulate predictions and labels
            all_preds.append(output.detach().cpu())
            all_labels.append(batch_y.detach().cpu())

    # concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # compute loss
    loss = loss_function(all_preds, all_labels)
    avg_loss = loss.item()

    # compute Pearson correlation
    epoch_corr = []
    if all_preds.dim() == 2 and all_preds.shape[1] > 1:
        # multi-output regression
        num_vars = all_preds.shape[1]
        for i in range(num_vars):
            corr, _ = pearsonr(all_preds[:, i], all_labels[:, i])
            epoch_corr.append(corr)
    else:
        # single-output regression
        corr, _ = pearsonr(all_preds.squeeze(), all_labels.squeeze())
        epoch_corr.append(corr)

    return avg_loss, epoch_corr

def test(loader, model, device):
    """
    Evaluate the model on test set and compute Pearson correlation.

    Args:
        loader: DataLoader for test set
        model: PyTorch model
        device: 'cuda' or 'cpu'

    Returns:
        epoch_corr: list of float, Pearson correlation per output variable
        all_labels: torch.Tensor, ground truth labels
        all_preds: torch.Tensor, model predictions
    """
    model.eval()  # set model to evaluation mode
    all_preds, all_labels = [], []

    with torch.no_grad():  # disable gradient computation
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            output = model(batch_x)

            all_preds.append(output.detach().cpu())
            all_labels.append(batch_y.detach().cpu())

    # concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # compute Pearson correlation
    epoch_corr = []
    if all_preds.dim() == 2 and all_preds.shape[1] > 1:
        # multi-output regression
        num_vars = all_preds.shape[1]
        for i in range(num_vars):
            corr, _ = pearsonr(all_preds[:, i], all_labels[:, i])
            epoch_corr.append(corr)
    else:
        # single-output regression
        corr, _ = pearsonr(all_preds.squeeze(), all_labels.squeeze())
        epoch_corr.append(corr)

    # epoch_corr = []
    # if all_preds.dim() == 2 and all_preds.shape[1] > 1:
    #     # multi-output regression
    #     num_vars = all_preds.shape[1]
    #     for i in range(num_vars):
    #         corr_parts = []
    #         preds_chunks = torch.chunk(all_preds[:, i], 5)
    #         labels_chunks = torch.chunk(all_labels[:, i], 5)
    #         for preds_part, labels_part in zip(preds_chunks, labels_chunks):
    #             corr, _ = pearsonr(preds_part.cpu().numpy(), labels_part.cpu().numpy())
    #             corr_parts.append(corr)
    #         epoch_corr.append(sum(corr_parts) / len(corr_parts))
    # else:
    #     # single-output regression
    #     corr_parts = []
    #     preds_chunks = torch.chunk(all_preds.squeeze(), 5)
    #     labels_chunks = torch.chunk(all_labels.squeeze(), 5)
    #     for preds_part, labels_part in zip(preds_chunks, labels_chunks):
    #         corr, _ = pearsonr(preds_part.cpu().numpy(), labels_part.cpu().numpy())
    #         corr_parts.append(corr)
    #     epoch_corr.append(sum(corr_parts) / len(corr_parts))

    return epoch_corr, all_labels, all_preds

class EarlyStopping_loss:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()
        else:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:

            return self.best_model_weights
        return None

class EarlyStopping_performance:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): 容忍多少个epoch没有提升
            min_delta (float): 至少要提升多少才算真正的提升
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('-inf')
        self.counter = 0
        self.best_model_weights = None

    def __call__(self, val_score, model):
        """
        Args:
            val_score (float): validation performance (accuracy/AUC/F1)
            model (torch.nn.Module):
        """
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_model_weights = model.state_dict()
        else:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            return self.best_model_weights
        return None

