import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pycppmsa_utils
import torch.nn.functional as F
import transformers
import numpy as np

class SyntacticTreeParser(nn.Module):
    def __init__(self, args, config):
        super(SyntacticTreeParser, self).__init__()
        self.args = args
        self.config = config
        if self.args.word_embedding_path:
            self.word_embdd = nn.Embedding.from_pretrained(torch.load(args.word_embedding_path), freeze=True)
        else:
            self.word_embdd = nn.Embedding(self.args.voc_size, self.config.word_embedding_dim)
        self.pos_embdd = nn.Embedding(self.args.nb_pos_tag, self.config.pos_embedding_dim)
        self.dropout_word_embdd = nn.Dropout(self.config.dropout)
        self.dropout_pos_embdd = nn.Dropout(self.config.dropout)
        self.lstm = nn.LSTM(self.config.lstm_hidden_size_in, self.config.lstm_hidden_size_out, batch_first=True, bidirectional=True, num_layers=self.config.lstm_num_layers, dropout=self.config.dropout)
        enc_output_dim = 2 * self.config.lstm_hidden_size_out

        self.mlp_arc_head = MLP(enc_output_dim, self.config.mlp_arc_head_dim, self.config.dropout)
        self.mlp_arc_dep = MLP(enc_output_dim, self.config.mlp_arc_dep_dim, self.config.dropout)

        self.mlp_lbl = MLP(enc_output_dim, self.config.mlp_lbl_dim, self.config.dropout)

        self.biaffine_arc = BiAffine(self.config.mlp_arc_head_dim, self.config.mlp_arc_dep_dim, bias_right=False, bias=False)
        self.biaffine_head = BiAffine(self.config.mlp_lbl_dim, self.config.mlp_lbl_dim, self.args.nb_lbl)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def _forward_rnn(self, input_ids, attention_mask, postag_ids):
        x = self.dropout_word_embdd(self.word_embdd(input_ids))
        y = self.dropout_pos_embdd(self.pos_embdd(postag_ids.squeeze()))
        x = torch.cat((x, y), dim=-1)
        # BxLx(2*E)
        sentences_length = attention_mask.sum(1)
        #x = nn.utils.rnn.pack_padded_sequence(x, sentences_length, batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)

        h_arc_head = self.mlp_arc_head(h) #B*L*H
        h_arc_dep = self.mlp_arc_dep(h) #B*L*H
        h_lbl = self.mlp_lbl(h) #B*L*H

        mask = attention_mask.unsqueeze(2)
        h_arc_head *= mask
        h_arc_dep *= mask
        h_lbl *= mask
        # BxLx(2*H)
        return h_arc_head, h_arc_dep, h_lbl

    def forward(self, input_ids, attention_mask, postag_ids, heads, labels):
        h_head, h_dep, h_lbl = self._forward_rnn(input_ids, attention_mask, postag_ids)
        scores_arc = self.biaffine_arc(h_head, h_dep) # B * 1 * L * L
        scores_lbl = self.biaffine_head(h_lbl, h_lbl) # B * O * L * L

        return self._loss(scores_arc, scores_lbl, attention_mask, heads, labels)

    def _loss(self, scores_arc, scores_lbl, attention_mask,  heads, labels):
        B, O, L, _, = scores_lbl.shape

        mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(B, 1, L, L)
        mask = mask * mask.transpose(-2, -1)
        scores_arc *= mask

        mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(B, O, L, L)
        mask = mask * mask.transpose(-2, -1)
        scores_lbl *= mask

        scores = scores_arc + scores_lbl
        scores = scores.transpose(1, -1)  # BxOxHxM -> BxMxHxO

        mask = torch.zeros(L)  # L*L
        mask.fill_(float("-inf"))
        mask = mask.diag()
        mask[0, :] = float("-inf")
        mask = mask.to(self.args.device)

        scores += mask.unsqueeze(-1)

        final_scores = scores[:, 1:].reshape(B, L - 1, L * O)

        heads_indices = heads * self.args.nb_lbl + labels.squeeze()
        loss = 0
        for i in range(B):
            loss += self.loss(
                final_scores[i],
                heads_indices[i, 1:]
            )
        res = loss/B
        if not self.training:
            #DUMMY
            #_, preds = torch.topk(final_scores, 1, dim=-1)
            #preds_head, preds_lbl = preds // self.args.nb_lbl, preds % self.args.nb_lbl
            #res = (res, preds_head.squeeze(), preds_lbl.squeeze(),)
            #MSA
            scores_arc = scores_arc.permute(0, 2, 3, 1) # B*1*H*M -> B*H*M*1
            scores_lbl = scores_lbl.permute(0, 2, 3, 1) # B*H*M*O -> B*H*M*O
            scores = scores_arc + scores_lbl
            W, all_arc_lbl = torch.max(scores, -1)
            W = W.cpu()
            all_arc_lbl = all_arc_lbl.cpu()

            all_preds_head = - torch.ones((B, L))
            all_preds_lbl = - torch.ones((B, L))
            lim = attention_mask.sum(-1).cpu().long()

            for i in range(B):
                w = W[i, :lim[i], :lim[i]]
                preds_head = pycppmsa_utils.as_heads(w, single_root=True, root_on_diag=False)
                all_preds_head[i, :lim[i]] = preds_head
                #all_preds_head.append(preds_head)

                arc_indices = preds_head[1:].unsqueeze(-1)
                preds_lbl = torch.gather((all_arc_lbl[i].T)[1:lim[i]], 1, arc_indices).squeeze()
                all_preds_lbl[i, 1:lim[i]] = preds_lbl


            res = (res, all_preds_head, all_preds_lbl,)

        return res


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super(MLP, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        return x


class BiAffine(nn.Module):

    def __init__(self, left_features_dim, right_features_dim, out_features_dim=1, bias_left=True, bias_right=True, bias=True):
        super(BiAffine, self).__init__()
        self.left_features_dim = left_features_dim
        self.right_features_dim = right_features_dim
        self.out_features_dim = out_features_dim
        self.bias_left = bias_left
        self.bias_right = bias_right
        self.bias = bias
        self.U = Parameter(torch.Tensor(out_features_dim, left_features_dim, right_features_dim))
        if bias_left:
            self.Wl = Parameter(torch.Tensor(self.out_features_dim, self.left_features_dim, 1))
        else:
            self.register_parameter('Wl', None)
        if bias_right:
            self.Wr = Parameter(torch.Tensor(self.out_features_dim, self.right_features_dim, 1))
        else:
            self.register_parameter('Wr', None)
        if bias:
            self.b = Parameter(torch.Tensor(out_features_dim))
        else:
            self.register_parameter('b', None)
        self.reset_paramaters()

    def reset_paramaters(self):
        nn.init.xavier_uniform(self.U)
        if self.bias_left:
            nn.init.xavier_uniform(self.Wl)
        if self.bias_right:
            nn.init.xavier_uniform(self.Wr)
        if self.bias:
            nn.init.constant_(self.b, 0.)

    def forward(self, input_left, input_right):
        b = input_left.shape[0]
        o = self.out_features_dim
        input_left = input_left.unsqueeze(1).expand(b, o, input_left.shape[1], input_left.shape[2])
        input_right = input_right.unsqueeze(1).expand(b, o, input_right.shape[1], input_right.shape[2])
        U = self.U.unsqueeze(0).expand(b, *self.U.shape)
        output = torch.matmul(torch.matmul(input_left, U), input_right.permute(0, 1, 3, 2))

        if self.bias_left:
            Wl = self.Wl.unsqueeze(0).expand(b, *self.Wl.shape)
            output += torch.matmul(input_left, Wl)
        if self.bias_right:
            Wr = self.Wr.unsqueeze(0).expand(b, *self.Wr.shape)
            output += torch.matmul(input_right, Wr)
        if self.bias:
            output += self.b.unsqueeze(0)\
                            .expand(b, *self.b.shape)\
                            .unsqueeze(-1)\
                            .unsqueeze(-1)
        return output
