import pickle

import torch
from torch import nn, device
from torch.utils import data

from heuristic_with_ranking.ranking_network.npuzzle_architecture import RankingNetworkNPuzzle
from heuristic_with_ranking.ranking_network.npuzzle_training import train_npuzzle_model

TRAINING_SET_LOCATION: str = 'training_examples_20000.pkl'
VALIDATION_SET_LOCATION: str = 'validation_examples_2000.pkl'
BATCH_SIZE: int = 128

def load_training_set(training_set_location: str,
                      validation_set_location: str,
                      batch_size: int = BATCH_SIZE) -> tuple[data.DataLoader, data.DataLoader]:
    with open(training_set_location, 'rb') as f:
        training_set = pickle.load(f)
    with open(validation_set_location, 'rb') as f:
        validation_set = pickle.load(f)
    training_loader: data.DataLoader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader: data.DataLoader = data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    return training_loader, validation_loader

def main():
    device_to_use: device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RankingNetworkNPuzzle(
        first_layer = (16, 256),
        other_layers = [512, 256, 128, 64, 32, 16]
    )
    model = model.to(device_to_use)
    training_loader, validation_loader = load_training_set(
        training_set_location=TRAINING_SET_LOCATION,
        validation_set_location=VALIDATION_SET_LOCATION,
    )
    returned_model, training_loss, validation_loss = train_npuzzle_model(
        training_set=training_loader,
        validation_set=validation_loader,
        model_to_train=model
    )



if __name__ == '__main__':
    main()