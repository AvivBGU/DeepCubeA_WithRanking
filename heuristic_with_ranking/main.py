import math

import torch
from numpy._typing import NDArray

from ctg_approx.avi import load_nnet
from environments.n_puzzle import NPuzzle, NPuzzleState
from utils import nnet_utils
from utils.nnet_utils import get_heuristic_fn

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


def load_pretrained_model(npuzzle_sqrt_dim: int) -> NPuzzle:
    if npuzzle_sqrt_dim != 4:
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






if __name__ == '__main__':
    load_pretrained_model(npuzzle_sqrt_dim=4)