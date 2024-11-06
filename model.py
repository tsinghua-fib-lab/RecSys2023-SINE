import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class MultiheadedAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super(MultiheadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.dropout_rate = dropout_rate

        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_V = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.fc = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, query, key, value, attn_mask, addition_intent_weights):
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)

        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask
        attn = attn + attn_mask
        p_attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_rate,
                           training=self.training)  # (128, 1, 200, 200) num_head:1
        if addition_intent_weights is not None:
            p_attn *= addition_intent_weights.to(p_attn.device)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)
        x = F.dropout(self.fc(x), p=self.dropout_rate, training=self.training)

        return x, p_attn


class SINE(torch.nn.Module):
    def __init__(self, item_num, new_maxlen, device, args):
        super(SINE, self).__init__()
        self.args = args
        self.item_num = item_num
        self.max_len = new_maxlen
        self.dev = device
        assert args.num_interest > 1, 'num_interest should be an integer greater than 1'
        self.num_interest = args.num_interest
        self.interest_embedding = torch.nn.Embedding(
            self.num_interest,
            args.hidden_units,
        )
        if args.adaptive:
            self.interest_attention_linear = torch.nn.Linear(args.hidden_units * 2, 1, bias=True)
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.max_len, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiheadedAttention(args.hidden_units,
                                                  args.num_heads,
                                                  args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, intent_weights):
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = torch.tile(torch.tensor(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]).to(torch.long)

        seqs += self.pos_emb(positions.to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask,
                                                      addition_intent_weights=intent_weights)

            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def project_k_interest(self, log_feats):
        interest_ids = torch.linspace(0, self.num_interest - 1, steps=self.num_interest, dtype=torch.long).to(self.dev)
        interest_ems = self.interest_embedding(interest_ids)
        # expand interest dim
        log_feats = log_feats.unsqueeze(2)
        log_feats = log_feats + torch.sigmoid(
            log_feats * interest_ems
        ) * log_feats
        return log_feats.permute(2, 0, 1, 3)

    def forward(self, log_feats, pos_seqs, neg_seqs, neg_feedback, filter):  # for training

        first_item_embs = self.item_emb(log_feats[:, 0].to(self.dev))
        first_item_embs = first_item_embs.unsqueeze(1)
        pos_embs = self.item_emb(pos_seqs.to(self.dev))  # (128, 200, 50)
        neg_embs = self.item_emb(neg_seqs.to(self.dev))
        fdbk_embs = self.item_emb(neg_feedback.to(self.dev))

        if filter:
            # pos_logits (3, 128, 200)
            interest_ids = torch.linspace(0, self.num_interest - 1, steps=self.num_interest, dtype=torch.long).to(
                self.dev)
            interest_ems = self.interest_embedding(interest_ids)  # (3, 50)
            pos_relations = (pos_embs.unsqueeze(2) * interest_ems).permute(2, 0, 1, 3).sum(dim=-1)
            fdbk_relations = (fdbk_embs.unsqueeze(2) * interest_ems).permute(2, 0, 1, 3).sum(dim=-1)
            first_item_relations = (first_item_embs.unsqueeze(2) * interest_ems).permute(2, 0, 1, 3).sum(dim=-1)
            fdbk_indices = torch.where(neg_feedback != 0, 1, 0).to(self.dev)
            fdbk_relations = fdbk_relations * fdbk_indices
            target_intent = torch.max(pos_relations - fdbk_relations, dim=0).indices  # (128, 200)
            first_item_intent = torch.max(first_item_relations, dim=0).indices  # (128, 1)
            input_intent = torch.cat((first_item_intent, target_intent[:, :-1]), dim=1)  # (128, 200)

            target_intent = target_intent.unsqueeze(1).repeat(1, target_intent.size(-1), 1).transpose(-2, -1)
            input_intent = input_intent.unsqueeze(1).repeat(1, input_intent.size(-1), 1)

            intent_weights = torch.tril((input_intent == target_intent) * filter)
            intent_weights = torch.where(intent_weights != filter, 1., filter).unsqueeze(1)
        else:
            intent_weights = None

        log_feats = self.log2feats(log_feats, intent_weights)
        log_feats = self.project_k_interest(log_feats)  # (3, 128, 200, 50)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (3, 128, 200)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        fdbk_logits = (log_feats * fdbk_embs).sum(dim=-1)

        if hasattr(self, 'interest_attention_linear'):
            pos_scores = F.softmax(self.interest_attention_linear(
                torch.cat(
                    (log_feats, pos_embs.unsqueeze(0).repeat(log_feats.size(0), 1, 1, 1)), dim=-1
                )
            ).squeeze(), dim=0)  # (3, 128, 200)
            neg_scores = F.softmax(self.interest_attention_linear(
                torch.cat(
                    (log_feats, neg_embs.unsqueeze(0).repeat(log_feats.size(0), 1, 1, 1)), dim=-1
                )
            ).squeeze(), dim=0)
            fdbk_scores = F.softmax(self.interest_attention_linear(
                torch.cat(
                    (log_feats, fdbk_embs.unsqueeze(0).repeat(log_feats.size(0), 1, 1, 1)), dim=-1
                )
            ).squeeze(), dim=0)

            pos_logits = (pos_logits * pos_scores).sum(dim=0)  # (128, 200)
            neg_logits = (neg_logits * neg_scores).sum(dim=0)
            fdbk_logits = (fdbk_logits * fdbk_scores).sum(dim=0)
            fdbk_logits = pos_logits - fdbk_logits
        else:
            diff = pos_logits - fdbk_logits
            pos_logits = pos_logits.sum(dim=0)
            neg_logits = neg_logits.sum(dim=0)
            fdbk_logits = torch.max(diff, dim=0).values

        return pos_logits, neg_logits, fdbk_logits

    @torch.no_grad()
    def predict(self, log_seqs, item_indices):  # for inference
        intent_weights = None
        log_feats = self.log2feats(log_seqs, intent_weights)  # [bs, len, 50]
        log_feats = self.project_k_interest(log_feats)
        final_feat = log_feats[:, :, -1, :]  # (3, bs, 50)

        item_embs = self.item_emb(item_indices.to(self.dev))  # (U, I, C) (bs, n, 50) target_item + candidate_item
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # (3, bs, n)

        if hasattr(self, 'interest_attention_linear'):
            item_scores = F.softmax(self.interest_attention_linear(
                torch.cat(
                    (final_feat.unsqueeze(2).repeat(1, 1, item_embs.size(1), 1),
                     item_embs.unsqueeze(0).repeat(final_feat.size(0), 1, 1, 1)),
                    dim=-1
                )
            ).squeeze(), dim=0)  # (3, bs, n)
            logits = (logits * item_scores).sum(dim=0)
        else:
            logits = logits.sum(dim=0)

        return logits  # preds # (U, I)
