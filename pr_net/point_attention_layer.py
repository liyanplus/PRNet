import torch as th
import torch.nn as th_nn

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class PointAttentionLayer(th_nn.Module):
    """
    docstring
    """

    def __init__(self, context_dim,
                 leaky_relu_slope=1e-2,
                 hidden_layer_num=4, hidden_feature_dim=256):
        super(PointAttentionLayer, self).__init__()

        self.linears = []
        self.linears.append(
            th_nn.Linear(context_dim, hidden_feature_dim).to(device)
        )
        for _ in range(2, hidden_layer_num):
            self.linears.append(
                th_nn.Linear(hidden_feature_dim, hidden_feature_dim).to(device)
            )
        self.linears.append(
            th_nn.Linear(hidden_feature_dim, 1).to(device)
        )
        self.activate = th_nn.LeakyReLU(leaky_relu_slope)
        self.weigh = th_nn.Sigmoid()

    def forward(self, context):
        """
        docstring
        """
        x = context
        for linear in self.linears[:-1]:
            x = self.activate(linear(x))
        return self.weigh(self.linears[-1](x))
