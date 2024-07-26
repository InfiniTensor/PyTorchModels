import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)

    def forward(self, x, hx):
        h, c = hx
        # Linear transformations
        gates = (torch.mm(x, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(h, self.weight_hh.t()) + self.bias_hh)

        # Split the gates into their respective components
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        # Apply nonlinearities
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        # Update cell state
        c_next = f_gate * c + i_gate * g_gate
        # Compute hidden state
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0):
        
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize LSTM layers
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else hidden_size * self.num_directions
            lstm_cell = CustomLSTMCell(input_dim, hidden_size, bias=bias)
            self.lstm_cells.append(lstm_cell)
        
        # Initialize projection layer if proj_size > 0
        if proj_size > 0:
            self.projection = nn.Linear(hidden_size * self.num_directions, proj_size, bias=bias)
        else:
            self.projection = None

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)  # Convert (N, L, Hin) to (L, N, Hin)

        seq_len, batch_size, _ = input.size()

        if hx is None:
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)
            c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=input.device)
        else:
            h_0, c_0 = hx

        h_n, c_n = [], []
        layer_output = input

        for layer in range(self.num_layers):
            h_t, c_t = h_0[layer], c_0[layer]
            outputs = []

            for t in range(seq_len):
                h_t, c_t = self.lstm_cells[layer](layer_output[t], (h_t, c_t))
                outputs.append(h_t)

            layer_output = torch.stack(outputs, dim=0)
            if self.bidirectional:
                h_t_rev, c_t_rev = h_0[layer + 1], c_0[layer + 1]
                outputs_rev = []

                for t in reversed(range(seq_len)):
                    h_t_rev, c_t_rev = self.lstm_cells[layer](layer_output[t], (h_t_rev, c_t_rev))
                    outputs_rev.append(h_t_rev)

                outputs_rev.reverse()
                layer_output = torch.cat((layer_output, torch.stack(outputs_rev, dim=0)), dim=2)

            h_n.append(h_t)
            c_n.append(c_t)

            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

        if self.projection is not None:
            layer_output = self.projection(layer_output)

        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)

        if self.batch_first:
            layer_output = layer_output.transpose(0, 1)  # Convert (L, N, Hout) back to (N, L, Hout)

        return layer_output, (h_n, c_n)