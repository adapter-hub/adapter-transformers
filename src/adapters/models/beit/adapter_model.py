from typing import Optional

import torch

from transformers.models.beit.modeling_beit import (
    BEIT_INPUTS_DOCSTRING,
    BEIT_START_DOCSTRING,
    BeitModel,
    BeitPreTrainedModel,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ...context import AdapterSetup
from ...heads import ModelWithFlexibleHeadsAdaptersMixin
from ...wrappers import init


@add_start_docstrings(
    """Beit Model transformer with the option to add multiple flexible heads on top.""",
    BEIT_START_DOCSTRING,
)
class BeitAdapterModel(ModelWithFlexibleHeadsAdaptersMixin, BeitPreTrainedModel):
    head_types = [
        "image_classification",
    ]
    use_pooler = True

    def __init__(self, config):
        super().__init__(config)

        self.beit = BeitModel(config)
        init(self.beit)

        self._init_head_modules()

        # Initialize weights and apply final processing
        self.post_init()

    # Overwrites the function from: transformers.modeling_utils.PreTrainedModel
    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings specifically for BEiT's tuple output format.
        """

        def make_inputs_require_grads(module, input, output):
            # >>> START AH Changes <<<
            # Handle BEiT's specific tuple output format. Hugging Face's implementation is buggy and doesn't work for BEiT.
            output[0].requires_grad_(True)
            # >>> END AH Changes <<<

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, context = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            output_context=True,
        )
        # required e.g. for prompt tuning in all models
        kwargs["context"] = context

        # BERT & RoBERTa return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                cls_output=pooled_output,  # BEiT does classification based on average-pooling of last hidden state
                head_name=head,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs
