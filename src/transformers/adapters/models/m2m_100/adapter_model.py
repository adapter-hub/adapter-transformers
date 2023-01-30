import warnings

import torch

from ....models.m2m_100.modeling_m2m_100 import (
    M2M_100_INPUTS_DOCSTRING,
    M2M_100_START_DOCSTRING,
    M2M100Config,
    M2M100Model,
    M2M100PreTrainedModel,
    shift_tokens_right,
)
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...composition import adjust_tensors_for_parallel
from ...heads import (
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    QuestionAnsweringHead,
    Seq2SeqLMHead,
)
from ...model_mixin import EmbeddingAdaptersWrapperMixin


@add_start_docstrings(
    "M2M100 Model with the option to add multiple flexible prediction heads on top.", M2M_100_START_DOCSTRING
)
class M2M100AdapterModel(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, M2M100PreTrainedModel):
    def __init__(self, config: M2M100Config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = M2M100Model(config)

        self._init_head_modules()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @add_start_docstrings_to_model_forward(M2M_100_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        head=None,
        output_adapter_gating_scores=False,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "labels" in kwargs or "start_positions" in kwargs and "end_positions" in kwargs:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            output_adapter_gating_scores=output_adapter_gating_scores,
        )
        # sequence classification based on last token in sequence
        x = outputs[0]  # last hidden state
        if input_ids is not None and x.shape[1] == input_ids.shape[1]:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            (eos_mask,) = adjust_tensors_for_parallel(x, eos_mask)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            cls_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        else:
            cls_representation = x

        head_outputs = self.forward_head(
            outputs,
            head_name=head,
            cls_output=cls_representation,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        return head_outputs

    # Copied from M2M100ForConditionalGeneration
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


    # Copied from M2M100ForConditionalGeneration
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    head_types = {
        "classification": ClassificationHead,
        "multilabel_classification": MultiLabelClassificationHead,
        "question_answering": QuestionAnsweringHead,
        "seq2seq_lm": Seq2SeqLMHead,
    }

    def add_classification_head(
        self,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        overwrite_ok=False,
        multilabel=False,
        id2label=None,
    ):
        """
        Adds a sequence classification head on top of the model.

        Args:
            head_name (str): The name of the head.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            layers (int, optional): Number of layers. Defaults to 2.
            activation_function (str, optional): Activation function. Defaults to 'tanh'.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
            multilabel (bool, optional): Enable multilabel classification setup. Defaults to False.
        """

        if multilabel:
            head = MultiLabelClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        else:
            head = ClassificationHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    def add_qa_head(
        self,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        overwrite_ok=False,
        id2label=None,
    ):
        head = QuestionAnsweringHead(self, head_name, num_labels, layers, activation_function, id2label)
        self.add_prediction_head(head, overwrite_ok)

    def add_seq2seq_lm_head(
        self,
        head_name,
        overwrite_ok=False,
    ):
        """
        Adds a sequence-to-sequence language modeling head on top of the model.

        Args:
            head_name (str): The name of the head.
            overwrite_ok (bool, optional): Force overwrite if a head with the same name exists. Defaults to False.
        """
        head = Seq2SeqLMHead(self, head_name)
        self.add_prediction_head(head, overwrite_ok=overwrite_ok)


class M2M100ModelWithHeads(M2M100AdapterModel):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                self.__class__.__bases__[0].__name__
            ),
            FutureWarning,
        )
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
