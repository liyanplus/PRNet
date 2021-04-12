import torch as th
import torch.nn as th_nn
from src.pr_net.pr_net import PRNet

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class PathologyClassifier(th_nn.Module):
    """
    docstring
    """

    def __init__(self, feature_type_count, grid_scale_count,
                 pr_representation_dim, pp_representation_dim,
                 apply_attention=True,
                 leaky_relu_slope=1e-2):
        super(PathologyClassifier, self).__init__()

        pr_representation_dim = grid_scale_count if pr_representation_dim is None else pr_representation_dim
        self.pr_net = PRNet(feature_type_count,
                            grid_scale_count,
                            pr_representation_dim,
                            pp_representation_dim,
                            apply_attention,
                            leaky_relu_slope)

        self.pr_representation_dim = pr_representation_dim
        self.classify_linear1 = th_nn.Linear(feature_type_count * feature_type_count * pr_representation_dim, 4096)
        self.classify_linear2 = th_nn.Linear(4096, 4096)
        self.classify_linear3 = th_nn.Linear(4096, 1)
        self.classify_activate = th_nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.final = th_nn.Sigmoid()

    def forward(self, raw_data,
                neighborhood_tensor,
                core_point_idxs):
        """
        docstring
        """
        pr_representations = self.pr_net(raw_data, neighborhood_tensor, core_point_idxs)

        x = self.classify_activate(self.classify_linear1(pr_representations))
        x = self.classify_activate(self.classify_linear2(x))
        return self.final(self.classify_linear3(x)), pr_representations
