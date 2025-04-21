from environments.n_puzzle import NPuzzle
from utils import env_utils
from utils.nnet_utils import get_heuristic_fn


def load_pretrained_model(nlight_puzzle_dim: int):
    nlights_puzzle: NPuzzle = NPuzzle(nlight_puzzle_dim)


if __name__ == '__main__':
    load_pretrained_model(15)
    print('hello world')