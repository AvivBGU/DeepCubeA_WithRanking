# Training should include a list of board states, for example 2/3/4 for n_puzzle,
# And the expected output should be valid ranking. A type of teacher forcing can be employed by requiring the
# Ranking to be precise.
# The pre-trained network of DeepCubeA will be used for bootstrapping to assess how good a situation is according
# to the distance to the goal by the helper network.
# Cases:
#    1. Same distance within some epsilon: Treat the states as the same.
#    2. Loss function? - MSE since this is a regression problem.
import functools
import os
import pickle
from typing import Callable

import numpy as np

from environments.environment_abstract import Environment
from environments.n_puzzle import NPuzzle, NPuzzleState
from heuristic_with_ranking.main import PRETRAINED_DIMS, load_pretrained_model

MAX_BACKWARD_STEPS: int = 10
STEPS_TO_GENERATE_AT_A_TIME: int = 1

def load_evaluator(environment: Environment) -> Callable:
    return load_pretrained_model()

def generate_training_set(numbers_of_example_to_generate: int,
                          generator: Callable,
                          evaluator: Callable) -> list[tuple[np.ndarray, float]]: # List of states of n-puzzle and eval.
    generated_examples: list[tuple[np.ndarray, float]] = list()
    for _ in range(numbers_of_example_to_generate):
        states, _ = generator()
        puzzle_state: NPuzzleState = states[0]
        example_evaluation = evaluator([puzzle_state])[-1]
        generated_examples.append((puzzle_state.tiles, example_evaluation))
    return generated_examples


def export_generated_examples(generated_examples: list[tuple[np.ndarray, float]], export_destination: str) -> None:
    updated_name = f"{export_destination}_{len(generated_examples)}.pkl"
    if os.path.exists(export_destination):
        print(f'File: {export_destination} exists, adding number to the end.')
        current_number: int = 1
        updated_name = f"{export_destination}_{len(generated_examples)}_{current_number}.pkl"
        while os.path.exists(updated_name):
            updated_name = f"{export_destination}_{len(generated_examples)}_{current_number}.pkl"
            current_number += 1
    with open(updated_name, 'wb') as f:
        pickle.dump(generated_examples, f)






if __name__ == '__main__':
    npuzzle_env: Environment = NPuzzle(PRETRAINED_DIMS)
    evaluator: Callable = load_evaluator(npuzzle_env)
    generator: Callable = functools.partial(
        npuzzle_env.generate_states,STEPS_TO_GENERATE_AT_A_TIME, (0, MAX_BACKWARD_STEPS)
    )
    generated_training_set = generate_training_set(2000,
        generator,
        evaluator
    )
    export_generated_examples(generated_training_set, "training_examples")