import torch as th
import torch.nn as th_nn
from src.pr_net.point_attention_layer import PointAttentionLayer
from src.pr_net.pointpair_representation_layer import PointpairRepresentationLayer
from src.pr_net.binary_context_layer import BinaryContextLayer

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class PRNet(th_nn.Module):
    """
    docstring
    """

    def __init__(self, feature_type_count, grid_scale_count,
                 pr_representation_dim, pp_representation_dim,
                 apply_attention=True,
                 leaky_relu_slope=1e-2):
        super(PRNet, self).__init__()

        self.feature_type_count = feature_type_count
        self.apply_attention = apply_attention
        self.pr_representation_dim = pr_representation_dim
        self.pointpair_representation_layer = PointpairRepresentationLayer(grid_scale_count,
                                                                           pp_representation_dim,
                                                                           leaky_relu_slope)
        self.binary_context_layer = BinaryContextLayer(feature_type_count)
        self.point_attention_layer = PointAttentionLayer(pp_representation_dim, leaky_relu_slope)

        self.linear = th_nn.Linear(pp_representation_dim, pr_representation_dim)
        self.activate = th_nn.LeakyReLU(leaky_relu_slope)

    def forward(self, raw_data, neighborhood_tensor, core_point_idxs):
        """
        docstring
        """
        # neighborhood_representation, pp_representation - selected point pairs
        pp_representation = self.pointpair_representation_layer(neighborhood_tensor)

        # binary_context
        raw_binary_context = self.binary_context_layer(
            raw_data, pp_representation, core_point_idxs
        )

        pr_reperesentation = th.zeros(
            self.feature_type_count ** 2 * self.pr_representation_dim, device=device)

        if self.apply_attention:
            # self_context - core points
            self_context = raw_binary_context[th.arange(raw_binary_context.shape[0]),
                           raw_data.long()[core_point_idxs], :]
            # point attention - core points
            point_attention = th.flatten(self.point_attention_layer(self_context)).to(device)
        else:
            point_attention = th.ones(th.Size((th.sum(core_point_idxs),)), device=device)

        binary_context = th.zeros(th.Size([
            th.sum(core_point_idxs),
            self.feature_type_count,
            self.pr_representation_dim]), device=device)

        # for feature_idx in range(self.feature_type_count):
        #     binary_context[:, feature_idx, :] = \
        #         self.activate(self.linear3(
        #             self.activate(self.linear2(
        #                 self.activate(self.linear1(raw_binary_context[:, feature_idx, :]))
        #             ))
        #         ))

        for feature_idx in range(self.feature_type_count):
            binary_context[:, feature_idx, :] = self.activate(self.linear(raw_binary_context[:, feature_idx, :]))

        for feature_idx in range(self.feature_type_count):
            if point_attention[raw_data[core_point_idxs] == feature_idx].shape[0] == 0 or \
                    th.sum(point_attention[raw_data[core_point_idxs] == feature_idx]) == 0:
                prs = th.zeros(self.feature_type_count * self.pr_representation_dim, device=device)
            else:
                if point_attention[raw_data[core_point_idxs] == feature_idx].shape[0] == 1:
                    prs = binary_context[raw_data[core_point_idxs] == feature_idx] * \
                          point_attention[raw_data[core_point_idxs] == feature_idx]
                else:
                    prs = th.sum(
                        th.mul(
                            binary_context[raw_data[core_point_idxs] == feature_idx],
                            point_attention[raw_data[core_point_idxs] == feature_idx].reshape((-1, 1, 1))
                        ),
                        dim=0
                    )
                prs = prs / th.sum(point_attention[raw_data[core_point_idxs] == feature_idx])
            pr_reperesentation[feature_idx * self.feature_type_count * self.pr_representation_dim:
                               (feature_idx + 1) * self.feature_type_count * self.pr_representation_dim] = prs.flatten()

        return pr_reperesentation
