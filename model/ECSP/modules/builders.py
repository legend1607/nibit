import torch
import torch.nn as nn

def build_shared_mlp(dims, conv_dim=1, conv_bias=True, act=nn.ReLU(inplace=True), final_act=False):
    '''
    default = n * (conv + bn + act) + (conv + bn)
    conv_dim: default=1; set conv_dim=2 if using grouped points (B, G, S, in_dim)
    to apply:
        if conv_dim=1:
            x = x.permute(0, 2, 1)  # (B, in_dim, N)
            x = self.mlp(x).max(dim=2)[0]  # (B, embed_dim)
        else:
            x = x.permute(0, 3, 2, 1)  # (B, in_dim, S, G)
            x = self.mlp(x).max(dim=2)[0]  # (B, embed_dim, S, G), max over neighborhood S -> (B, embed_dim, G) 
            x = x.permute(0, 2, 1)  # (B, G, embed_dim)
    '''
    assert conv_dim in [1, 2], "conv_dim must be 1 or 2"
    conv = nn.Conv1d if conv_dim == 1 else nn.Conv2d
    bn = nn.BatchNorm1d if conv_dim == 1 else nn.BatchNorm2d

    layers = []
    n_layers = len(dims) - 1

    for i in range(n_layers):
        in_dim, out_dim = dims[i], dims[i + 1]
        is_last = i == n_layers - 1

        layers.append(conv(in_dim, out_dim, kernel_size=1, bias=conv_bias))
        layers.append(bn(out_dim))
        if not is_last or final_act:
            layers.append(act)

    return nn.Sequential(*layers)


def build_fc_layers(dims, linear_bias=True, bn=True, act=nn.ReLU(inplace=True), dropout=0.0):
    '''
    apply linear projection along last dim
    default = n * (linear + bn + act + dropout) + linear
    dropout: float or list of float(s), dropout rate(s) per layer
    '''
    layers = []
    n_layers = len(dims) - 1

    # Normalize dropout to list of length (n_layers - 1)
    if isinstance(dropout, (float, int)):
        dropout = [dropout] * (n_layers - 1)
    assert len(dropout) == n_layers - 1, f"Expected dropout list of length {n_layers - 1}"

    for i in range(n_layers):
        in_dim, out_dim = dims[i], dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim, bias=linear_bias))
        
        if i < n_layers - 1:  # Not the final layer
            if bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act)
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
                
    return nn.Sequential(*layers)
