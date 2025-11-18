import functools
import pickle
import tkinter as tk
from typing import Callable

import numpy as np
import torch
from torch import device
from torch.utils import data
from torchsummary import summary

from heuristic_with_ranking.display_npuzzle import PuzzleGUI
from heuristic_with_ranking.ranking_network.npuzzle_architecture import RankingNetworkNPuzzle
from heuristic_with_ranking.ranking_network.npuzzle_training import train_npuzzle_model

TRAINING_SET_LOCATION: str = 'training_examples_20000.pkl'
VALIDATION_SET_LOCATION: str = 'validation_examples_2000.pkl'
BATCH_SIZE: int = 128

def cost_to_goal_ranking(state: np.ndarray, heuristic_func: Callable) -> float:
    with torch.no_grad():
        ret_vals = heuristic_func(state)
        return np.float64(ret_vals.item())

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
        other_layers = [512, 256, 128, 64, 32, 16],
        batchnorm_between_layers = False,
    )
    model = model.to(device_to_use)
    summary(model, input_size=(16, ), batch_size=128)
    training_loader, validation_loader = load_training_set(
        training_set_location=TRAINING_SET_LOCATION,
        validation_set_location=VALIDATION_SET_LOCATION,
    )
    returned_model, training_loss, validation_loss = train_npuzzle_model(
        training_set=training_loader,
        validation_set=validation_loader,
        model_to_train=model,
        device_to_use=device_to_use
    )
    returned_model.eval()
    root = tk.Tk()
    root.title("15 Puzzle")

    start = np.array([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 0,
        13, 14, 15, 12
    ])

    cost_to_goal_with_model_loaded = functools.partial(cost_to_goal_ranking, heuristic_func=returned_model)

    app = PuzzleGUI(root, start, cost_to_goal_with_model_loaded)
    print("About to enter main tkinter loop")
    root.mainloop()



if __name__ == '__main__':
    main()