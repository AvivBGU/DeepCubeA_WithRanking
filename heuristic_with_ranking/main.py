import math
import pickle
import tkinter as tk
from multiprocessing import Queue
from typing import Callable

import numpy as np

import torch
from numpy._typing import NDArray

from ctg_approx.avi import load_nnet
from environments.environment_abstract import Environment
from environments.n_puzzle import NPuzzle, NPuzzleState
from scripts.generate_dataset import generate_and_save_states
from utils import nnet_utils
from utils.nnet_utils import get_heuristic_fn

PRETRAINED_DIMS: int = 4
EXAMPLES_TO_GENERATE: int = 10
STATES_TO_GENERATE: int = 10
MAX_NUMBERS_OF_BACKWARDS_MOVES_TO_GENERATE: int = 30

########## GUI #############

class PuzzleGUI:
    def __init__(self, root, initial_state: np.ndarray, heuristic_function: Callable):
        self.root = root
        self.state = initial_state.copy()
        self.buttons = [[None]*4 for _ in range(4)]

        self.status = tk.Label(root, text="", font=("Courier", 12))
        self.status.grid(row=4, column=0, columnspan=4, pady=(10, 0))

        self.entry = tk.Entry(root, width=35, font=("Courier", 12))
        self.entry.grid(row=5, column=0, columnspan=3, pady=10, padx=5)
        self.load_button = tk.Button(root, text="Load State", font=("Courier", 12), command=self.load_state)
        self.load_button.grid(row=5, column=3, padx=5)
        self.heuristic_function = heuristic_function
        self.draw_board()

    def draw_board(self):
        for i in range(4):
            for j in range(4):
                val = self.state[i*4 + j]
                text = "" if val == 0 else str(val)
                if not self.buttons[i][j]:
                    btn = tk.Button(self.root, text=text, width=4, height=2,
                                    font=("Courier", 16, "bold"),
                                    command=lambda x=i, y=j: self.move_tile(x, y))
                    btn.grid(row=i, column=j, padx=5, pady=5)
                    self.buttons[i][j] = btn
                else:
                    self.buttons[i][j]['text'] = text
        self.update_status()

    def cost_to_goal(self, state: np.ndarray, heuristic_function: Callable) -> int:
        puzzle_as_state: NPuzzleState = NPuzzleState(state)
        return heuristic_function([puzzle_as_state])[-1]

    def update_status(self, msg=None):
        cost = self.cost_to_goal(self.state, self.heuristic_function)
        self.status['text'] = msg or f"Estimated Cost to Goal: {cost}"

    def move_tile(self, x, y):
        zero_idx = np.where(self.state == 0)[0][0]
        zx, zy = divmod(zero_idx, 4)
        if abs(zx - x) + abs(zy - y) == 1:
            idx1, idx2 = zx * 4 + zy, x * 4 + y
            self.state[idx1], self.state[idx2] = self.state[idx2], self.state[idx1]
            self.draw_board()

    def load_state(self):
        try:
            values = list(map(int, self.entry.get().split(',')))
            if sorted(values) != list(range(16)):
                raise ValueError("Invalid tile values")
            self.state = np.array(values)
            self.draw_board()
        except Exception as e:
            self.update_status(f"Error loading state: {e}")

########## GUI END #########

def get_device():
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))
    return device


def print_npuzzle_board(np_puzzle_board_state: NPuzzleState):
    da_tiles: NDArray = np_puzzle_board_state.tiles
    dimension: int = int(math.sqrt(len(da_tiles)))
    for cell_num in range(da_tiles.shape[0]):
        if cell_num % dimension == 0 and cell_num:
            print()
        print(f'{da_tiles[cell_num]:3d} ', end='')


def load_pretrained_model(npuzzle_sqrt_dim: int = PRETRAINED_DIMS) -> Callable:
    if npuzzle_sqrt_dim != PRETRAINED_DIMS:
        raise ValueError('npuzzle_sqrt_dim must be 4, will be added later.')
    nlights_puzzle: NPuzzle = NPuzzle(npuzzle_sqrt_dim)
    device_to_run = get_device()
    npuzzle_dir: str = r'C:\Users\Avivm\Desktop\Thesis_work\DeepCubeA_WithRanking\saved_models\puzzle15\target'
    neural_net_model, iter, update_num = load_nnet(nnet_dir=npuzzle_dir, env=nlights_puzzle)
    heuristic_fn = get_heuristic_fn(
        nnet = neural_net_model,
        device = device_to_run,
        env = nlights_puzzle
    )
    return heuristic_fn

def load_npuzzle(pickled_problem_path: str, heuristic_function: Callable) -> list[int]:
    data = pickle.load(open(pickled_problem_path, "rb"))
    for single_puzzle in data['states']:
        single_puzzle: NPuzzle

        root = tk.Tk()
        root.title("15 Puzzle")

        app = PuzzleGUI(root, single_puzzle.tiles, heuristic_function=heuristic_function)
        root.mainloop()

def generate_examples(examples_to_generate: int = EXAMPLES_TO_GENERATE,
                      environment: Environment = None):
    if environment is None:
        environment = NPuzzle(PRETRAINED_DIMS)
    file_queue: Queue = Queue()
    file_queue.put(r'C:\Users\Avivm\Desktop\Thesis_work\DeepCubeA_WithRanking\heuristic_with_ranking\generated_puzzles\generation_test.pkl')
    generate_and_save_states(
        env = environment,
        num_states = examples_to_generate,
        back_max = MAX_NUMBERS_OF_BACKWARDS_MOVES_TO_GENERATE,
        filepath_queue = file_queue
    )





if __name__ == '__main__':
    # generate_examples()
    load_npuzzle(pickled_problem_path='generated_puzzles/generation_test.pkl',
                 heuristic_function=load_pretrained_model())