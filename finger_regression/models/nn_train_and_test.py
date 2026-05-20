import torch
import copy
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

    return epoch_corr, all_labels, all_preds

class EarlyStopping_performance:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('-inf')
        self.counter = 0
        self.best_model_weights = None
        self.best_epoch = 0

    def __call__(self, val_score, model, epoch):
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            # print(f"Best model updated at epoch {epoch} with score {val_score:.4f}")
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False

    def load_best_model(self, model):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            print(f"Loaded best model from epoch {self.best_epoch} (Score: {self.best_score:.4f})")
