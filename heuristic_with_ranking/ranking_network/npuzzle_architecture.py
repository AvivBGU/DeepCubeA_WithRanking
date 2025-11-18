import numpy as np
import torch
import torch.nn as nn

class RankingNetworkNPuzzle(nn.Module):
    # Input should be the board state. The best move should receive the highest number.
    def __init__(self,
                 first_layer: tuple[int, int], # In/out dims
                 other_layers: list[int], # / Out dims
                 batchnorm_between_layers: bool = True):
        super().__init__()
        layers = [nn.Linear(in_features=first_layer[0],
                            out_features=first_layer[1])]
        if batchnorm_between_layers:
            layers.append(nn.BatchNorm1d(first_layer[1]))
        layers.append(nn.LeakyReLU())
        prev_layer_out: int = first_layer[1]
        for layer_dim in other_layers:
            layers.append(nn.Linear(in_features=prev_layer_out,
                                          out_features=layer_dim))
            if batchnorm_between_layers:
                layers.append(nn.BatchNorm1d(layer_dim))
            layers.append(nn.LeakyReLU())
            prev_layer_out: int = layer_dim

        layers.append(nn.Linear(in_features=prev_layer_out,
                                      out_features=1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)


    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            if isinstance(x, list):
                # Convert list of arrays to a single ndarray first
                x = np.asarray(x, dtype=np.float32)

            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            x = x.float().to(next(self.parameters()).device)
        return self.network(x)
