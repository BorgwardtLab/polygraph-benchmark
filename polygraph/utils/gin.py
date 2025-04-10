"""
Based on https://github.com/uoguelph-mlrg/GGM-metrics, modified to use torch_geometric but identical computation-wise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.conv import MessagePassing


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network layer implemented in PyTorch Geometric
    Closely mirrors the original DGL implementation
    """

    def __init__(
        self, apply_func, aggregator_type, init_eps=0, learn_eps=False
    ):
        # Determine reducer based on aggregator type
        if aggregator_type == "sum":
            self._reducer = "add"
        elif aggregator_type == "max":
            self._reducer = "max"
        elif aggregator_type == "mean":
            self._reducer = "mean"
        else:
            raise KeyError(f"Aggregator type {aggregator_type} not recognized.")

        # Initialize message passing with message aggegration type
        super().__init__(aggr=self._reducer)

        self.apply_func = apply_func
        self._aggregator_type = aggregator_type

        # Handle epsilon parameter
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        # Handle optional edge weight
        if edge_weight is not None:
            assert False

        # Propagate messages with optional edge attributes
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr
        )
        assert out.ndim == x.ndim
        diff = out.shape[-1] - x.shape[-1]
        if diff != 0:
            zeros = torch.zeros((*x.shape[:-1], diff)).to(x.device)
            padded_x = torch.cat([x, zeros], dim=-1)
        else:
            padded_x = x

        # Apply epsilon-weighted update
        rst = (1 + self.eps) * padded_x + out

        # Apply optional MLP function
        if self.apply_func is not None:
            rst = self.apply_func(rst)

        return rst

    def message(self, x_j, edge_weight=None, edge_attr=None):
        # Default message
        m = x_j

        # Optional edge weight multiplication
        if edge_weight is not None:
            m = m * edge_weight.view(-1, 1)

        # Optional edge attribute concatenation
        if edge_attr is not None:
            m = torch.cat([m, edge_attr], dim=-1)

        return m


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)

        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""

    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        graph_pooling_type,
        neighbor_pooling_type,
        edge_feat_dim=0,
        final_dropout=0.0,
        learn_eps=False,
        output_dim=1,
        seed=None,
        **kwargs,
    ):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """

        super().__init__()

        if seed is not None:
            if kwargs["init"] != "orthogonal":
                raise ValueError(
                    "Seeding has no effect on non-orthogonal initialization."
                )

            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        def init_weights_orthogonal(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, generator=generator)
                # Use standard initialization for bias
                if m.bias is not None and generator is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound, generator=generator)

            elif isinstance(m, MLP):
                if hasattr(m, "linears"):
                    m.linears.apply(init_weights_orthogonal)
                else:
                    m.linear.apply(init_weights_orthogonal)
            elif isinstance(m, nn.ModuleList):
                pass
            else:
                raise Exception()

        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(
                    num_mlp_layers,
                    input_dim + edge_feat_dim,
                    hidden_dim,
                    hidden_dim,
                )
            else:
                mlp = MLP(
                    num_mlp_layers,
                    hidden_dim + edge_feat_dim,
                    hidden_dim,
                    hidden_dim,
                )
            if kwargs["init"] == "orthogonal":
                init_weights_orthogonal(mlp)

            self.ginlayers.append(
                GINConv(
                    ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim)
                )

        if kwargs["init"] == "orthogonal":
            logger.debug("Initializing GIN linear layers orthogonally")
            self.linears_prediction.apply(init_weights_orthogonal)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = global_add_pool
        elif graph_pooling_type == "mean":
            self.pool = global_mean_pool
        elif graph_pooling_type == "max":
            self.pool = global_max_pool
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, batch, edge_attr=None):
        # list of hidden representation at each layer (including input)
        hidden_rep = [x]

        # h = self.preprocess_nodes(h)
        for i in range(self.num_layers - 1):
            x = self.ginlayers[i](x, edge_index, edge_attr=edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            hidden_rep.append(x)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(x=h, batch=batch)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer

    def get_graph_embed(self, x, edge_index, batch, edge_attr=None):
        self.eval()
        with torch.no_grad():
            # return self.forward(g, h).detach().numpy()
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                x = self.ginlayers[i](x, edge_index, edge_attr=edge_attr)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                hidden_rep.append(x)

            # perform pooling over all nodes in each graph in every layer
            graph_embed = torch.Tensor([]).to(x.device)
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(x=h, batch=batch)
                graph_embed = torch.cat([graph_embed, pooled_h], dim=1)

            return graph_embed

    def get_graph_embed_no_cat(self, x, edge_index, batch, edge_attr=None):
        self.eval()
        with torch.no_grad():
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                x = self.ginlayers[i](x, edge_index, edge_attr=edge_attr)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                hidden_rep.append(x)

            return self.pool(x=hidden_rep[-1], batch=batch)
