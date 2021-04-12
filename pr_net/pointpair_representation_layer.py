import torch as th
import torch.nn as th_nn

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class PointpairRepresentationLayer(th_nn.Module):
    """
    docstring
    """

    def __init__(self, grid_scale_count, pointpair_representation_dim,
                 leaky_relu_slope=1e-2,
                 hidden_layer_num=4, hidden_feature_dim=256):
        super(PointpairRepresentationLayer, self).__init__()
        self.pointpair_representation_dim = pointpair_representation_dim
        self.linears = []
        self.linears.append(
            th_nn.Linear(6 * grid_scale_count, hidden_feature_dim).to(device)
        )
        for _ in range(2, hidden_layer_num):
            self.linears.append(
                th_nn.Linear(hidden_feature_dim, hidden_feature_dim).to(device)
            )
        self.linears.append(
            th_nn.Linear(hidden_feature_dim, pointpair_representation_dim).to(device)
        )
        self.activate = th_nn.LeakyReLU(leaky_relu_slope).to(device)

    def forward(self, neighborhood_tensor):
        """
        docstring
        """
        x = neighborhood_tensor._values()
        for linear in self.linears:
            x = self.activate(linear(x))
        return th.sparse_coo_tensor(neighborhood_tensor._indices(), x,
                                    th.Size([neighborhood_tensor.shape[0],
                                             neighborhood_tensor.shape[0],
                                             self.pointpair_representation_dim])
                                    )
