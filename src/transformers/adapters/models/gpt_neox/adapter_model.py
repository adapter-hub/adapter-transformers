import logging

import torch

from ....models.gpt_neox.modeling_gpt_neox import GPT_NEOX_START_DOCSTRING, GPTNeoXModel, GPTNeoXPreTrainedModel
from ....utils import add_start_docstrings
from ...composition import adjust_tensors_for_parallel
from ...heads import CausalLMHead, ModelWithFlexibleHeadsAdaptersMixin
from ...model_mixin import EmbeddingAdaptersWrapperMixin


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
The GPTNeoX Model that allows the loading of different heads for different tasks. This enables a flexible use of the
models and adapters. Since this class does classification on the last token, it requires to know the position of the
last token. If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding
token in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same
(take the last value in each row of the batch).
""",
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXAdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)

        self._init_head_modules()

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
        )

        batch_size = outputs[0].shape[0]

        if self.config.pad_token_id is None:
            # TODO-AH: this may result in unexpected behavior for classification. Find a better way to do this?
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                (sequence_lengths,) = adjust_tensors_for_parallel(outputs[0], sequence_lengths)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        cls_logits = outputs[0][range(batch_size), sequence_lengths]

        outputs = self.forward_head(
            outputs,
            head_name=head,
            cls_output=cls_logits,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs

    # Copied from GPTNeoXForCausalLM
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    head_types = {"causal_lm": CausalLMHead}

    def add_causal_lm_head(self, head_name, overwrite_ok=False):
        """
        Adds a causal language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = CausalLMHead(self, head_name)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)
