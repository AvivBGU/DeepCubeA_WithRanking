import copy
import time

import torch
from torch import nn
from torch.utils import data

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
PATIENCE: int = 10
MAX_EPOCHS: int = 2000
PATIENCE_EXPECTED: float = 1e-3

def train_npuzzle_model(model_to_train: nn.Module,
                        training_set: data.DataLoader,
                        validation_set: data.DataLoader,
                        device_to_use: torch.device):
    optimizer = torch.optim.Adam(
                                 model_to_train.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY
                                 )
    criterion = torch.nn.MSELoss()
    current_validation_loss: float = 100000.0
    train_loss_per_epoch: list[float] = list()
    validation_loss_per_epoch: list[float] = list()
    patience_for_improvement: float = 0
    time_for_all_training = time.time()
    current_epoch: int = 0
    best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
    for epoch in range(MAX_EPOCHS):
        running_train_loss = 0.0
        running_validation_loss = 0.0
        model_to_train.train()
        start = time.time()
        for batch, targets in training_set:
            optimizer.zero_grad()
            batch_in_device = batch.to(device_to_use)
            normalized_targets = 1 - (targets + 1)/81 # TODO It's hacky but should work. Need to figure out smarter solution.
            targets = normalized_targets.view(-1, 1)
            targets_device = targets.to(device_to_use)
            model_output = model_to_train(batch_in_device)
            train_loss = criterion(model_output, targets_device)
            running_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
        avg_train_loss = running_train_loss / len(training_set)
        train_loss_per_epoch.append(avg_train_loss)
        model_to_train.eval()
        with torch.no_grad():
            for validation_batch, validation_targets in validation_set:
                validation_batch = validation_batch.to(device_to_use)
                validation_model_output = model_to_train(validation_batch)
                normalized_validation_targets = 1 - (validation_targets + 1) / 81 # TODO It's hacky but should work.
                validation_targets = normalized_validation_targets.view(-1, 1)
                validation_targets_in_device = validation_targets.to(device_to_use)
                validation_loss = criterion(validation_model_output, validation_targets_in_device)
                running_validation_loss += validation_loss.item()
            avg_validation_loss = running_validation_loss / len(validation_set)
            validation_loss_per_epoch.append(avg_validation_loss)
        if current_validation_loss - avg_validation_loss < PATIENCE_EXPECTED:
            # Performance degraded
            patience_for_improvement += 1
            performance_degraded = True
        else:
            patience_for_improvement = 0
            current_validation_loss = avg_validation_loss
            best_model_state_dict = copy.deepcopy(model_to_train.state_dict())
            performance_degraded = False
        elapsed_time = time.time() - start
        print(f'Epoch {epoch + 1}/{MAX_EPOCHS} completed in {elapsed_time:.2f}. Avg training loss: {avg_train_loss} Avg validation loss: {avg_validation_loss}')
        current_epoch += 1
        if performance_degraded:
            print(f'No significant improvement. Patience factor: {patience_for_improvement}/{PATIENCE}')
        if patience_for_improvement >= PATIENCE:
            print('No significant improvement. Patience factor reached. Stopping training.')
            break
    print(f'Finished in {time.time() - time_for_all_training:.2f} seconds, {current_epoch} epochs')
    model_to_train.load_state_dict(best_model_state_dict)
    model_to_train.eval()
    return model_to_train, train_loss_per_epoch, validation_loss_per_epoch,

