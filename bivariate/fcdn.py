import torch
import torch.nn as nn

# Note that while we use blind denoiser setup here to be consistent with the rest of the paper,
# a non-blind denoiser can potentially perform much better with the 2D data (future work).
class Denoiser(nn.Module):
    '''
    A fully connected denoiser for bivariate input.
    '''
    def __init__(self, n_node, n_int, bias=False) -> None:
        super().__init__()

        # Setup the network
        inter = []
        for i in range(n_int):
            inter.append(nn.Linear(n_node, n_node, bias=bias))
            inter.append(nn.ReLU())

        self.model = nn.Sequential(
            nn.Linear(2, n_node, bias=bias),
            nn.ReLU(),
            *inter,
            nn.Linear(n_node, 2, bias=bias))

        # Initialize the weights
        self._weight_init()

    def forward(self, x):
        return self.model(x)

    def score(self, x):
        with torch.no_grad():
            return - self.model(x)

    def _weight_init(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(init_func)