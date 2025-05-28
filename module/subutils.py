import torch
from torch.nn.utils.rnn import pack_padded_sequence


def seq2seq_forward(encoder, decoder, lstm_input, valid_len, pre_len):
    his_len = (valid_len - pre_len).to('cpu')
    src_padded_embed = pack_padded_sequence(lstm_input, his_len, batch_first=True, enforce_sorted=False)
    _, hc = encoder(src_padded_embed)
    trg_embed = torch.stack([torch.cat([lstm_input[i, start - 1:start], lstm_input[i, -pre_len:-1]], dim=0)
                             for i, start in enumerate(his_len)], dim=0)
    decoder_out, _ = decoder(trg_embed, hc)  # (batch_size, pre_len, hidden_size)
    return decoder_out


def rnn_forward(encoder, sos, lstm_input, valid_len, pre_len):
    batch_size = lstm_input.size(0)
    input_size = lstm_input.size(-1)
    history_len = valid_len - pre_len
    max_len = history_len.max()

    lstm_input = torch.cat([sos.reshape(1, 1, -1).repeat(batch_size, 1, 1), lstm_input], dim=1)  # (batch, seq_len+1, 1+input_size)
    lstm_input = torch.stack([torch.cat([lstm_input[i, :s + 1], lstm_input[i, -pre_len:-1],
                                         torch.zeros(max_len - s, input_size).float().to(lstm_input.device)], dim=0)
                              for i, s in enumerate(history_len)], dim=0)
    lstm_out, _ = encoder(lstm_input)
    lstm_out_pre = torch.stack([lstm_out[i, s - pre_len:s] for i, s in enumerate(valid_len)])  # (batch, pre_len, lstm_hidden_size)
    return lstm_out_pre
