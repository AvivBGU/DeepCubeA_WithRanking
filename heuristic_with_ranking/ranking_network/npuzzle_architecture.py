import torch.nn as nn

class RankingNetworkNPuzzle(nn.Module):

    # Input should be the board state. The best move should receive the highest number.
    def __init__(self,
                 first_layer: tuple[int, int], # In/out dims
                 other_layers: tuple[int], # / Out dims
                 batchnorm_between_layers: bool = True):
        super().__init__()
        self.network = nn.Sequential()
        self.network.append(nn.Linear(in_features=first_layer[0],
                                      out_features=first_layer[1])
                            )
        if batchnorm_between_layers:
            self.network.append(nn.BatchNorm1d(first_layer[1]))
        self.network.append(nn.LeakyReLU())
        prev_layer_out: int = first_layer[1]
        for layer_dim in other_layers:
            self.network.append(nn.Linear(in_features=prev_layer_out,
                                          out_features=layer_dim))
            if batchnorm_between_layers:
                self.network.append(nn.BatchNorm1d(layer_dim))
            self.network.append(nn.LeakyReLU())
            prev_layer_out: int = layer_dim

        self.network.append(nn.Linear(in_features=prev_layer_out,
                                      out_features=1))
        self.network.append(nn.Sigmoid())

    def forward(self, x):
        return self.network(x)

