import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class TGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean', bias=True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin_root = Linear(in_channels, out_channels, bias=bias)
        self.lin_l = Linear(in_channels, out_channels, bias=bias)
        self.lin_t = Linear(1, out_channels, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_root.reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_t.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight):
        out = self.propagate(edge_index, x=x, edge_attr=edge_weight)
        x_r = self.lin_root(x)
        out += x_r
        return out
    
    def message(self, x_j, edge_attr):
        neighbor_message = self.lin_l(x_j)
        time_message = self.lin_t(edge_attr.view(-1, 1))
        return neighbor_message + time_message
    
    def __repr__(self):
        return '{}({}, {}, aggr={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.aggr)
        