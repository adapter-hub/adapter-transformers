# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DeBERTa-v2 model."""

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Output,
    DebertaV2SelfOutput,
    DisentangledSelfAttention,
    scaled_size_sqrt,
)

from ...composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from ...utils import prefix_attention_mask
from ..bert.mixin_bert import BertOutputAdaptersMixin, BertSelfOutputAdaptersMixin
from .mixin_deberta_v2 import DebertaV2SelfAttentionAdaptersMixin


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutputWithAdapters(BertSelfOutputAdaptersMixin, DebertaV2SelfOutput):
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class DebertaV2OutputWithAdapters(BertOutputAdaptersMixin, DebertaV2Output):
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states


class DisentangledSelfAttentionWithAdapters(DebertaV2SelfAttentionAdaptersMixin, DisentangledSelfAttention):
    def transpose_for_scores_extended(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.ByteTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].
        """
        # >>> START AH Changes <<<
        attention_mask = prefix_attention_mask(attention_mask, dim=3, prefix_value=1)  # type: ignore
        attention_mask = prefix_attention_mask(attention_mask, dim=2, prefix_value=1)  # type: ignore
        # >>> END AH Changes <<<

        if query_states is None:
            query_states = hidden_states

        # >>> START AH Changes <<<
        query_layer = self.transpose_for_scores_extended(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores_extended(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores_extended(self.value_proj(hidden_states), self.num_attention_heads)

        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(query_layer, key_layer, value_layer)
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)

        orig_key_layer = key_layer.contiguous()  # save this for relative attention
        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask, False
        )  # [:, 0, :, 0])
        query_layer, orig_key_layer = adjust_tensors_for_parallel(key_layer, query_layer, orig_key_layer)

        query_layer = query_layer.contiguous().view(-1, query_layer.size(2), query_layer.size(-1))
        key_layer = key_layer.contiguous().view(-1, key_layer.size(2), key_layer.size(-1))
        value_layer = value_layer.contiguous().view(-1, value_layer.size(2), value_layer.size(-1))
        orig_key_layer = orig_key_layer.contiguous().view(-1, orig_key_layer.size(2), orig_key_layer.size(-1))
        # >>> END AH Changes <<<

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(query_layer, scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            # >>> START AH Changes <<<
            rel_att = self.disentangled_attention_bias(
                query_layer, orig_key_layer, relative_pos, rel_embeddings, scale_factor
            )
            # >>> END AH Changes <<<

        if rel_att is not None:
            # >>> START AH Changes <<<
            # rel_att is set to 0 by default, i.e. rel_att is always not None (don't know why HuggingFace does this).
            # Hence, we must check whether rel_att is a tensor and if so, pad it with zeros to be able to add it to attention_scores.
            if isinstance(rel_att, torch.Tensor):
                rel_att_padded = torch.zeros_like(attention_scores)
                rel_att_padded[:, :, -rel_att.size(2) :] = rel_att
                attention_scores = attention_scores + rel_att_padded
            else:
                attention_scores = attention_scores + rel_att
            # >>> END AH Changes <<<

        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        )

        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        # bsz x height x length x dimension
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs.masked_fill(attention_mask, 0)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if not output_attentions:
            return (context_layer, None)
        return (context_layer, attention_probs)
