import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch import nn
from torch.nn.utils.rnn import PackedSequence

__all__ = ['RNNStack']

def safe_pack_padded_sequence(input_tensor, lengths, batch_first=True):
    """
    Packs a padded sequence into a PackedSequence object.
    Ensures data is structured correctly (timestep by timestep).
    """
    # Ensure lengths is a 1D tensor on the same device as input_tensor
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=input_tensor.device)
    else:
        lengths = lengths.to(input_tensor.device)

    # Sort sequences by length in descending order
    lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
    # If lengths are on CPU from torch.sort, move sorted_indices to input_tensor.device
    # (Typically, torch.sort output device matches input, but good to be explicit if issues arise)
    sorted_indices = sorted_indices.to(input_tensor.device) 
    
    input_sorted = input_tensor[sorted_indices] if batch_first else input_tensor[:, sorted_indices]

    # Calculate batch_sizes (number of active sequences at each time step)
    # Ensure lengths_sorted is on CPU for this list comprehension if it involves .item() or comparison with Python int
    # Or, perform operations on GPU and then move to CPU list
    max_len = lengths_sorted[0].item()
    batch_sizes_list = []
    if lengths_sorted.device.type != 'cpu': # Move to CPU for safe iteration and .sum().item()
        lengths_sorted_cpu = lengths_sorted.cpu()
        batch_sizes_list = [(lengths_sorted_cpu > t).sum().item() for t in range(max_len)]
    else:
        batch_sizes_list = [(lengths_sorted > t).sum().item() for t in range(max_len)]
    
    # print("Generated batch_sizes (list):", batch_sizes_list) # Debug output

    # Construct the 'data' tensor correctly:
    # Concatenate elements timestep by timestep for active sequences
    data_parts = []
    for t_step in range(max_len):
        num_active_sequences_at_t = batch_sizes_list[t_step]
        if batch_first:
            # input_sorted has shape (batch, seq_len, features)
            active_data_at_t = input_sorted[:num_active_sequences_at_t, t_step, :]
        else:
            # input_sorted has shape (seq_len, batch, features)
            active_data_at_t = input_sorted[t_step, :num_active_sequences_at_t, :]
        data_parts.append(active_data_at_t)

    if not data_parts: # Handle empty input
        feature_dim = input_tensor.size(-1)
        data = torch.empty(0, feature_dim, device=input_tensor.device, dtype=input_tensor.dtype)
    else:
        data = torch.cat(data_parts, dim=0)

    # batch_sizes tensor for PackedSequence should be on CPU as per some PyTorch internals/expectations
    batch_sizes_tensor = torch.tensor(batch_sizes_list, dtype=torch.int64, device='cpu')

    return PackedSequence(
        data=data,
        batch_sizes=batch_sizes_tensor, # Standard practice is CPU tensor for batch_sizes
        sorted_indices=sorted_indices,
        unsorted_indices=torch.argsort(sorted_indices)
    )

def safe_pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    """
    Manually implements pad_packed_sequence logic, adapting to the PackedSequence format.
    Args:
        sequence: PackedSequence object
        batch_first: Output是否为 batch_first 格式
        padding_value: 填充值
        total_length: 强制填充到的长度（若不为None）
    Returns:
        padded_sequence: 填充后的张量
        lengths: 实际长度列表 (in original batch order)
    """
    data = sequence.data
    # batch_sizes should be a CPU tensor, convert to list
    batch_sizes_list = sequence.batch_sizes.cpu().numpy().tolist()
    sorted_indices = sequence.sorted_indices # On the same device as original input
    unsorted_indices = sequence.unsorted_indices # On the same device as original input

    # Determine maximum time step for the padded output
    actual_max_len_in_pack = len(batch_sizes_list)
    if total_length is None:
        max_len_output = actual_max_len_in_pack
    else:
        max_len_output = max(actual_max_len_in_pack, total_length)

    # Original batch size (number of sequences)
    original_batch_size = batch_sizes_list[0] if batch_sizes_list else 0
    if original_batch_size == 0: # Handle empty sequence
        feature_dim = data.size(-1) if data.numel() > 0 else 0
        output_shape = (original_batch_size, max_len_output, feature_dim) if batch_first else (max_len_output, original_batch_size, feature_dim)
        return torch.full(output_shape, padding_value, dtype=data.dtype, device=data.device), torch.zeros(original_batch_size, dtype=torch.int64, device=data.device)


    feature_dim = data.shape[1]
    
    # Initialize padded_output for sorted sequences first
    if batch_first:
        # Shape: (original_batch_size, max_len_output, feature_dim)
        padded_output_sorted = torch.full((original_batch_size, max_len_output, feature_dim), padding_value, dtype=data.dtype, device=data.device)
    else:
        # Shape: (max_len_output, original_batch_size, feature_dim)
        padded_output_sorted = torch.full((max_len_output, original_batch_size, feature_dim), padding_value, dtype=data.dtype, device=data.device)

    # Fill data into padded_output_sorted
    ptr = 0
    for t in range(actual_max_len_in_pack): # Iterate up to actual content length
        current_batch_size_at_t = batch_sizes_list[t]
        if current_batch_size_at_t == 0: # Should not happen if actual_max_len_in_pack > 0
            continue
        
        segment = data[ptr : ptr + current_batch_size_at_t]
        if batch_first:
            padded_output_sorted[:current_batch_size_at_t, t] = segment
        else:
            padded_output_sorted[t, :current_batch_size_at_t] = segment
        ptr += current_batch_size_at_t

    # Unsort padded_output to original batch order
    if unsorted_indices is not None:
        if batch_first:
            padded_output_final = padded_output_sorted.index_select(0, unsorted_indices)
        else:
            padded_output_final = padded_output_sorted.index_select(1, unsorted_indices)
    else: # Already in original order (e.g. batch_size=1 or already sorted input)
        padded_output_final = padded_output_sorted
        
    # Calculate actual lengths of sequences in sorted order first
    lengths_sorted_list = [0] * original_batch_size
    for t_idx in range(actual_max_len_in_pack):
        current_bs_at_t = batch_sizes_list[t_idx]
        for i in range(current_bs_at_t):
            lengths_sorted_list[i] += 1
    
    lengths_sorted_tensor = torch.tensor(lengths_sorted_list, dtype=torch.int64, device=data.device)

    # Unsort lengths to match original batch order
    if unsorted_indices is not None:
        final_lengths = lengths_sorted_tensor.index_select(0, unsorted_indices)
    else:
        final_lengths = lengths_sorted_tensor
            
    return padded_output_final, final_lengths

class RNNForward(nn.Module):
    def __init__(self, rnn_input_size, h_size, use_gru):
        super().__init__()
        if use_gru:
            self.rnn = nn.GRU(input_size=rnn_input_size,
                              hidden_size=h_size,
                              bidirectional=False,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=rnn_input_size,
                               hidden_size=h_size,
                               bidirectional=False,
                               batch_first=True)
        self.norm = nn.LayerNorm(h_size)

    def forward(self, x, x_lens, init_state):
        # x = nn.utils.rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True)
        x = safe_pack_padded_sequence(x, x_lens)
        x, final_state = self.rnn(x, init_state)  # [B, T, D]
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x, _ = safe_pad_packed_sequence(x, batch_first=True)
        x = self.norm(x)
        return x, final_state
    
    
class RNNStack(nn.Module):
    """堆叠单向GRU层

    :param i_size: GRU层的输入大小
    :type i_size: int
    :param h_size: GRU层的隐层大小
    :type h_size: int
    :param num_rnn_layers: rnn层数
    :type num_rnn_layers: int
    :param use_gru: 使用使用GRU，否则使用LSTM
    :type use_gru: bool

    :return: RNN组的输出层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, num_rnn_layers: int, use_gru: bool):
        super().__init__()
        self.rnns = nn.ModuleList()
        self.output_dim = h_size
        self.num_rnn_layers = num_rnn_layers
        self.use_gru = use_gru
        self.rnns.append(RNNForward(rnn_input_size=i_size, h_size=h_size, use_gru=use_gru))
        for i in range(0, self.num_rnn_layers - 1):
            self.rnns.append(RNNForward(rnn_input_size=h_size, h_size=h_size, use_gru=use_gru))

    def forward(self, x, x_lens, init_state_h_box=None, init_state_c_box=None):
        if init_state_h_box is not None:
            if self.use_gru is True:
                init_state_h_list = torch.split(init_state_h_box, self.num_rnn_layers, dim=0)
                init_state_list = init_state_h_list
            else:
                init_state_h_list = torch.split(init_state_h_box, self.num_rnn_layers, dim=0)
                init_state_c_list = torch.split(init_state_c_box, self.num_rnn_layers, dim=0)
                init_state_list = [(init_state_h_list[i], init_state_c_list[i]) for i in range(self.num_rnn_layers)]
        else:
            init_state_list = [None] * self.num_rnn_layers
        final_chunk_state_list = []
        for rnn, init_state in zip(self.rnns, init_state_list):
            x, final_state = rnn(x, x_lens, init_state)
            final_chunk_state_list.append(final_state)

        if self.use_gru is True:
            final_chunk_state_h_box = torch.concat(final_chunk_state_list, dim=0)
            final_chunk_state_c_box = init_state_c_box
        else:
            final_chunk_state_h_list = [final_chunk_state_list[i][0] for i in range(self.num_rnn_layers)]
            final_chunk_state_c_list = [final_chunk_state_list[i][1] for i in range(self.num_rnn_layers)]
            final_chunk_state_h_box = torch.concat(final_chunk_state_h_list, dim=0)
            final_chunk_state_c_box = torch.concat(final_chunk_state_c_list, dim=0)
        return x, final_chunk_state_h_box, final_chunk_state_c_box
