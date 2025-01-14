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
"""PyTorch DeBERTa model."""

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.deberta.modeling_deberta import (
    DebertaEmbeddings,
    DebertaOutput,
    DebertaSelfOutput,
    DisentangledSelfAttention,
    scaled_size_sqrt,
)

from ...composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from ...utils import prefix_attention_mask
from ..bert.mixin_bert import BertOutputAdaptersMixin, BertSelfOutputAdaptersMixin
from .mixin_deberta import DebertaSelfAttentionAdaptersMixin


class DebertaSelfOutputWithAdapters(BertSelfOutputAdaptersMixin, DebertaSelfOutput):
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states


class DebertaOutputWithAdapters(BertOutputAdaptersMixin, DebertaOutput):
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states


class DebertaEmbeddingsWithAdapters(DebertaEmbeddings):
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            # >>> START AH Changes <<<
            # HuggingFace uses += instead of + which leads to a bug when using model.enable_input_require_grads. Once this is fixed, we can remove
            embeddings = embeddings + position_embeddings
            # >>> END AH Changes <<<
        if self.token_type_embeddings is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            # >>> START AH Changes <<<
            embeddings = embeddings + token_type_embeddings
            # >>> END AH Changes <<<
        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DisentangledSelfAttentionWithAdapters(DebertaSelfAttentionAdaptersMixin, DisentangledSelfAttention):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """

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
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:
            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            q = torch.matmul(qkvw[0], query_states.t().to(dtype=qkvw[0].dtype))
            k = torch.matmul(qkvw[1], hidden_states.t().to(dtype=qkvw[1].dtype))
            v = torch.matmul(qkvw[2], hidden_states.t().to(dtype=qkvw[2].dtype))
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        # >>> START AH Changes <<<
        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(query_layer, key_layer, value_layer)
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)
        # >>> END AH Changes <<<

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        # >>> START AH Changes <<<
        orig_key_layer = key_layer  # save this for relative attention
        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask, False
        )
        (query_layer, orig_key_layer) = adjust_tensors_for_parallel(key_layer, query_layer, orig_key_layer)
        # >>> END AH Changes <<<

        rel_att: int = 0
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = scaled_size_sqrt(query_layer, scale_factor)
        query_layer = query_layer / scale.to(dtype=query_layer.dtype)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.relative_attention and rel_embeddings is not None and relative_pos is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            # >>> START AH Changes <<<
            rel_att = self.disentangled_att_bias(
                query_layer, orig_key_layer, relative_pos, rel_embeddings, scale_factor
            )
            # >>> END AH Changes <<<

        if rel_att is not None:
            # >>> START AH Changes <<<
            # rel_att is set to 0 by default, i.e. rel_att is always not None (don't know why HuggingFace does this).
            # Hence, we must check whether rel_att is a tensor and if so, pad it with zeros to be able to add it to attention_scores.
            if isinstance(rel_att, torch.Tensor):
                rel_att_padded = torch.zeros_like(attention_scores)
                rel_att_padded[:, :, :, -rel_att.size(-1) :] = rel_att
                attention_scores = attention_scores + rel_att_padded
            else:
                attention_scores = attention_scores + rel_att
            # >>> END AH Changes <<<

        # bxhxlxd
        if self.head_logits_proj is not None:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        # bsz x height x length x dimension
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs.masked_fill(attention_mask, 0)

        attention_probs = self.dropout(attention_probs)
        if self.head_weights_proj is not None:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if not output_attentions:
            return (context_layer, None)
        return (context_layer, attention_probs)
