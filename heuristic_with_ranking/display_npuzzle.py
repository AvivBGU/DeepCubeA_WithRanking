import tkinter as tk

import numpy as np

from ctg_approx.avi import load_nnet
from environments.n_puzzle import NPuzzle, NPuzzleState
from heuristic_with_ranking.main import get_device
from utils.nnet_utils import get_heuristic_fn

nlights_puzzle: NPuzzle = NPuzzle(4)
device_to_run = get_device()
npuzzle_dir: str = r'C:\Users\Avivm\Desktop\Thesis_work\DeepCubeA_WithRanking\saved_models\puzzle15\target'
neural_net_model, iter, update_num = load_nnet(nnet_dir=npuzzle_dir, env=nlights_puzzle)
heuristic_fn = get_heuristic_fn(
        nnet = neural_net_model,
        device = device_to_run,
        env = nlights_puzzle
    )

# Replace this with your actual cost function
def cost_to_goal(state: np.ndarray) -> int:
    puzzle_as_state: NPuzzleState = NPuzzleState(state)
    return heuristic_fn([puzzle_as_state])[-1]

class PuzzleGUI:
    def __init__(self, root, initial_state: np.ndarray):
        self.root = root
        self.state = initial_state.copy()
        self.buttons = [[None]*4 for _ in range(4)]

        self.status = tk.Label(root, text="", font=("Courier", 12))
        self.status.grid(row=4, column=0, columnspan=4, pady=(10, 0))

        self.entry = tk.Entry(root, width=35, font=("Courier", 12))
        self.entry.grid(row=5, column=0, columnspan=3, pady=10, padx=5)
        self.load_button = tk.Button(root, text="Load State", font=("Courier", 12), command=self.load_state)
        self.load_button.grid(row=5, column=3, padx=5)

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

    def update_status(self, msg=None):
        cost = cost_to_goal(self.state)
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

if __name__ == "__main__":
    root = tk.Tk()
    root.title("15 Puzzle")

    start = np.array([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 0,
        13, 14, 15, 12
    ])

    app = PuzzleGUI(root, start)
    root.mainloop()