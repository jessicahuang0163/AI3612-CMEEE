from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput


@dataclass
class NEROutputs(ModelOutput):
    """
    NOTE: `logits` here is the CRF decoding result.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.LongTensor] = None


def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=True,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver'):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            if self.conditional:
                if self.hidden_units is not None:
                    if self.hidden_initializer == 'normal':
                        torch.nn.init.normal(self.hidden_dense.weight)
                    elif self.hidden_initializer == 'xavier':
                        torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
                if self.center:
                    torch.nn.init.constant_(self.beta_dense.weight, 0)
                if self.scale:
                    torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)

        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class W2NERDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            label_num: int,
            dropout: float,  # for interface compatibility
            lstm_hid_size: int = 512,
            dist_emb_size: int = 20,
            type_emb_size: int = 20,
            emb_dropout: float = 0.5,
            conv_hid_size: int = 96,
            dilation: List[int] = [1, 2, 3],
            conv_dropout: float = 0.5,
            biaffine_size: int = 512,
            ffnn_hid_size: int = 288,
            out_dropout: float = 0.33):
        super().__init__()
        self.lstm_hid_size = lstm_hid_size
        self.conv_hid_size = conv_hid_size

        self.dis_embs = nn.Embedding(20, dist_emb_size)
        self.reg_embs = nn.Embedding(3, type_emb_size)

        lstm_input_size = hidden_size
        self.encoder = nn.LSTM(lstm_input_size, lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)

        conv_input_size = lstm_hid_size + dist_emb_size + type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, conv_hid_size, dilation, conv_dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.predictor = CoPredictor(label_num, lstm_hid_size, biaffine_size,
                                     conv_hid_size * len(dilation), ffnn_hid_size,
                                     out_dropout)
        self.cln = LayerNorm(lstm_hid_size, lstm_hid_size, conditional=True)

        self.loss_fct = nn.CrossEntropyLoss()

        self.adamp_weights = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def forward(self, hidden_states, grid_mask2d, dist_inputs, sent_length, labels: Optional[torch.Tensor] = None,
                no_decode: bool = False) -> NEROutputs:
        # discard CLS and SEP tokens
        word_reps = hidden_states[:, 1:-1, :]
        assert word_reps.size(1) == sent_length.max()

        # LSTM encoding
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)

        loss, logits = None, None
        if labels is not None:
            grid_mask_ = grid_mask2d.clone()
            loss = self.loss_fct(outputs[grid_mask_], labels[grid_mask_])

            if not no_decode:
                logits = outputs.argmax(dim=-1)
        else:
            logits = outputs.argmax(dim=-1)

        return NEROutputs(loss, logits)


class BertForW2NER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = W2NERDecoder(config.hidden_size, num_labels1, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            grid_mask2d=None,
            dist_inputs=None,
            length=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        sequence_output = torch.stack(sequence_output[2][-4:], dim=-1).mean(-1)

        output = self.classifier.forward(sequence_output, grid_mask2d, dist_inputs, length, labels, no_decode=no_decode)
        return output
