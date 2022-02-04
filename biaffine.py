# stolen from https://github.com/yzhangcs/biaffine-parser/tree/master/parser
class BatchedBiaffine(nn.Module):
    def __init__(self, input_dim, proj_dim, n_labels=1, output_bias=True, activation="tanh", bias_x=True, bias_y=True, dropout=0, negative_slope=0.1):
        super(BatchedBiaffine, self).__init__()

        self.head_projection = MLP(input_dim, proj_dim, activation=activation, dropout=dropout, negative_slope=negative_slope)
        self.mod_projection = MLP(input_dim, proj_dim, activation=activation, dropout=dropout, negative_slope=negative_slope)

        self.n_in = input_dim
        self.n_out = n_labels
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_labels, proj_dim + bias_x, proj_dim + bias_y))

        if output_bias:
            self.output_bias = nn.Parameter(torch.Tensor(1, 1, 1, n_labels))
        else:
            self.output_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.output_bias is not None:
            nn.init.zeros_(self.output_bias)

    def forward(self, features):
        x = self.head_projection(features)
        y = self.mod_projection(features)

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, seq_len, seq_len, n_out]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)

        if self.output_bias is not None:
            s = s + self.output_bias

        return s