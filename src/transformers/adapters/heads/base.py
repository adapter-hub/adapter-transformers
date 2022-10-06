import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import (
    ImageClassifierOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...utils import ModelOutput
from ..composition import AdapterCompositionBlock, BatchSplit, Parallel, parse_heads_from_composition
from ..context import AdapterSetup, ForwardContext
from ..model_mixin import ModelWithHeadsAdaptersMixin
from ..modeling import Activation_Function_Class


logger = logging.getLogger(__name__)


@dataclass
class MultiHeadOutput(ModelOutput):
    head_outputs: List[ModelOutput] = None
    loss: Optional[torch.FloatTensor] = None

    def __getitem__(self, k):
        # with number indices the head output at that position is accessed
        # e.g output[1] is equivalent to output.head_outputs[1]
        if isinstance(k, int):
            return self.head_outputs[k]
        # with strings the attribute in the underlying dict can be adressed
        # e.g output["loss"] is equivalent to output.loss
        else:
            return super().__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, int):
            self.head_outputs[k] = v
        else:
            return super().__setitem__(k, v)

    def __iter__(self):
        # iterates over the head outputs
        return iter(self.head_outputs)

    def __len__(self):
        return len(self.head_outputs)


# Let this class inherit from nn.Sequential to provide iterable access as before
class PredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config
        pred_head = []
        dropout_prob = self.config.get("dropout_prob", model_config.hidden_dropout_prob)
        bias = self.config.get("bias", True)
        for l_id in range(self.config["layers"]):
            if dropout_prob > 0:
                pred_head.append(nn.Dropout(dropout_prob))
            if l_id < self.config["layers"] - 1:
                pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
                if self.config["activation_function"]:
                    pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            else:
                if "num_labels" in self.config:
                    pred_head.append(nn.Linear(model_config.hidden_size, self.config["num_labels"], bias=bias))
                elif "num_choices" in self.config:  # used for multiple_choice head
                    pred_head.append(nn.Linear(model_config.hidden_size, 1, bias=bias))
                else:
                    pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=bias))
                    if self.config["activation_function"]:
                        pred_head.append(Activation_Function_Class(self.config["activation_function"]))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        return None  # override for heads with output embeddings

    def get_label_names(self):
        return ["labels"]


class ClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class MultiLabelClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
        bias=True,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "multilabel_classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            if labels.dtype != torch.float32:
                labels = labels.float()
            loss = loss_fct(logits, labels)

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqSequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class MultipleChoiceHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_choices=2,
        layers=2,
        activation_function="tanh",
        id2label=None,
        use_pooler=False,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "multiple_choice",
            "num_choices": num_choices,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=None, **kwargs):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
        logits = logits.view(-1, self.config["num_choices"])
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if return_dict:
            return MultipleChoiceModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class TaggingHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        id2label=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "tagging",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        logits = super().forward(outputs[0])
        loss = None

        labels = kwargs.pop("labels", None)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config["num_labels"])
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class QuestionAnsweringHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=1,
        activation_function="tanh",
        id2label=None,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "question_answering",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        sequence_output = outputs[0]
        logits = super().forward(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_positions = kwargs.pop("start_positions", None)
        end_positions = kwargs.pop("end_positions", None)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if return_dict:
            if isinstance(outputs, Seq2SeqModelOutput):
                return Seq2SeqQuestionAnsweringModelOutput(
                    loss=total_loss,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
            else:
                return QuestionAnsweringModelOutput(
                    loss=total_loss,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            outputs = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            if total_loss is not None:
                outputs = (total_loss,) + outputs
            return outputs

    def get_label_names(self):
        return ["start_positions", "end_positions"]


class ImageClassificationHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        num_labels=2,
        layers=2,
        activation_function="tanh",
        multilabel=False,
        id2label=None,
        use_pooler=False,
        bias=True,
    ):
        super().__init__(head_name)
        self.config = {
            "head_type": "image_classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "multilabel": multilabel,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label is not None else None,
            "use_pooler": use_pooler,
            "bias": bias,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        if labels is not None:
            if self.config["num_labels"] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config["multilabel"]:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config["num_labels"]), labels.view(-1))

        if return_dict:
            return ImageClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = (logits,) + outputs[1:]
            if labels is not None:
                outputs = (loss,) + outputs
            return outputs


class ModelWithFlexibleHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    """
    Adds flexible prediction heads to a model class. Implemented by the XModelWithHeads classes.
    """

    head_types: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_to_flex_head = True
        if not hasattr(self.config, "custom_heads"):
            self.config.custom_heads = {}
        self._active_heads = []

    def _init_head_modules(self):
        # this dict is _only_ used for saving & reloading the configs and should not be modified otherwise
        if not hasattr(self.config, "prediction_heads"):
            self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        for head_name, config in self.config.prediction_heads.items():
            self.add_prediction_head_from_config(head_name, config)

    # The following methods are required for handling LM heads

    def get_output_embeddings(self):
        # Only gets the output embeddings for the currently active head
        if self.active_head in self.heads:
            head = self.heads[self.active_head]
            return head.get_output_embeddings()
        else:
            return None

    def set_output_embeddings(self, new_embeddings):
        # Only sets the output embeddings for the currently active head
        if self.active_head in self.heads:
            head = self.heads[self.active_head]
            if head.get_output_embeddings() is not None:
                head.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        for head_name, head in self.heads.items():
            output_embeddings = head.get_output_embeddings()
            if output_embeddings is not None and self.config.tie_word_embeddings:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)
        
        return self.get_input_embeddings()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if not self.config.tie_word_embeddings:
            for head in self.heads.values():
                old_lm_head = self.get_output_embeddings()
                if old_lm_head is not None:
                    new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
                    self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    # Methods for managing prediction heads

    def add_prediction_head_from_config(
        self,
        head_name: str,
        config: dict,
        overwrite_ok: bool = False,
        set_active: bool = True,
    ):
        head_type = config.pop("head_type")
        # handle cases when id2label, label2id or both are available
        id2label = config.pop("id2label", None)
        if id2label is None:
            label2id = config.pop("label2id", None)
            if label2id is not None:
                id2label = {id_: label for label, id_ in label2id.items()}
        else:
            # don't pass label2id to head_class
            config.pop("label2id", None)
        # re-add id2label map to config
        if id2label is not None:
            config["id2label"] = id2label

        if head_type in self.head_types:
            head_class = self.head_types[head_type]
            head = head_class(self, head_name, **config)
            self.add_prediction_head(head, overwrite_ok=overwrite_ok, set_active=set_active)
        elif head_type in self.config.custom_heads:
            # we have to re-add the head type for custom heads
            self.add_custom_head(head_type, head_name, overwrite_ok=overwrite_ok, **config)
        else:
            raise AttributeError(
                "Given head type '{}' is not known. Please register this head type before loading the model".format(
                    head_type
                )
            )

    def get_prediction_heads_config(self):
        heads = {}
        for head_name, head in self.heads.items():
            heads[head_name] = head.config
        return heads

    def register_custom_head(self, identifier, head):
        self.config.custom_heads[identifier] = head

    @property
    def active_head(self) -> Union[str, List[str]]:
        """
        The active prediction head configuration of this model. Can be either the name of a single available head
        (string) or a list of multiple available heads. In case of a list of heads, the same base model is forwarded
        through all specified heads.

        Returns:
            Union[str, List[str]]: A string or a list of strings describing the active head configuration.
        """
        if not self._active_heads:
            return None
        elif len(self._active_heads) == 1:
            return self._active_heads[0]
        else:
            return self._active_heads

    @active_head.setter
    def active_head(self, head_name_or_list: Union[str, List[str], AdapterCompositionBlock]):
        if isinstance(head_name_or_list, str):
            if head_name_or_list and head_name_or_list not in self.heads:
                raise ValueError(f"Model does not contain a head with name '{head_name_or_list}'.")
            self._active_heads = [head_name_or_list] if head_name_or_list else None
            # If we set a single head, also switch the label mapping. For multiple head, that doesn't make sense?
            if head_name_or_list:
                self.config.label2id = self.heads[head_name_or_list].config.get("label2id", None)
                self.config.id2label = self.get_labels_dict(head_name_or_list)
        else:
            self._active_heads = head_name_or_list

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. This setting can be overriden by passing
        the `adapter_names` parameter in the `foward()` pass. If no adapter with the given name is found, no module of
        the respective type will be activated. In case the calling model class supports named prediction heads, this
        method will attempt to activate a prediction head with the name of the last adapter in the list of passed
        adapter names.

        Args:
            adapter_setup (list):
                The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        self.base_model.set_active_adapters(adapter_setup, skip_layers)
        # use last adapter name as name of prediction head
        if self.active_adapters:
            head_setup = parse_heads_from_composition(self.active_adapters)
            if isinstance(head_setup, str):
                head_setup = [head_setup]
            if head_setup and all(head in self.heads for head in head_setup):
                self.active_head = head_setup
            else:
                logger.info(
                    "Could not identify valid prediction head(s) from setup '{}'.".format(self.active_adapters)
                )

    def add_custom_head(self, head_type, head_name, overwrite_ok=False, set_active=True, **kwargs):
        if head_type in self.config.custom_heads:
            head = self.config.custom_heads[head_type](self, head_name, **kwargs)
            # When a build-in head is added as a custom head it does not have the head_type property
            if not hasattr(head.config, "head_type"):
                head.config["head_type"] = head_type
            self.add_prediction_head(head, overwrite_ok, set_active=set_active)
        else:
            raise AttributeError(
                "The given head as a head_type that is not registered as a custom head yet."
                " Please register the head first."
            )

    def add_prediction_head(
        self,
        head: PredictionHead,
        overwrite_ok: bool = False,
        set_active: bool = True,
    ):
        if head.name not in self.heads or overwrite_ok:
            self.heads[head.name] = head
            # add reference to model config to save all head configs too
            self.config.prediction_heads[head.name] = head.config

            # Set a default label2id map if not given
            if "label2id" in head.config.keys() and head.config["label2id"] is None:
                if "num_labels" in head.config.keys():
                    head.config["label2id"] = {"LABEL_" + str(num): num for num in range(head.config["num_labels"])}
                if "num_choices" in head.config.keys():
                    head.config["label2id"] = {"LABEL_" + str(num): num for num in range(head.config["num_choices"])}

            # In case the added head has tied weights, tie them here.
            self.tie_weights()

            logger.info(f"Adding head '{head.name}' with config {head.config}.")
            if set_active:
                self.active_head = head.name

        else:
            raise ValueError(
                f"Model already contains a head with name '{head.name}'. Use overwrite_ok=True to force overwrite."
            )

    def delete_head(self, head_name: str):
        """
        Deletes the prediction head with the specified name from the model.

        Args:
            head_name (str): The name of the prediction to delete.
        """
        if head_name not in self.config.prediction_heads:
            logger.info("No prediction head '%s' found for deletion. Skipping.", head_name)
            return
        del self.config.prediction_heads[head_name]
        del self.heads[head_name]
        if self.active_head == head_name:
            self.active_head = None

    def forward_head(
        self, all_outputs, head_name=None, cls_output=None, attention_mask=None, return_dict=False, **kwargs
    ):
        """
        The forward pass through a prediction head configuration. There are three ways to specify the used prediction
        head configuration (in order of priority):

            1. If a head_name is passed, the head with the given name is used.
            2. If the forward call is executed within an ``AdapterSetup`` context, the head configuration is read from
               the context.
            3. If the ``active_head`` property is set, the head configuration is read from there.

        Args:
            all_outputs (dict): The outputs of the base model.
            head_name (str, optional): The name of the prediction head to use. If None, the active head is used.
            cls_output (torch.Tensor, optional): The classification output of the model.
            attention_mask (torch.Tensor, optional): The attention mask of the model.
            return_dict (bool): Whether or not to return a ``ModelOutput`` instead of a plain tuple.
            **kwargs: Additional keyword arguments passed to the forward pass of the head.
        """
        if head_name:
            used_heads = [head_name]
        elif AdapterSetup.get_context_head_setup():
            used_heads = AdapterSetup.get_context_head_setup()
            if isinstance(used_heads, str):
                used_heads = [used_heads]
        elif self._active_heads:
            used_heads = self._active_heads
        else:
            logger.debug("No prediction head is used.")
            return all_outputs

        def _get_head_input(outputs, cls_out, batch):
            # TODO-AH check possible edge cases here
            if isinstance(outputs, ModelOutput):
                inputs = {}
                for key, base_output in outputs.items():
                    if torch.is_tensor(base_output):
                        inputs[key] = base_output[batch[0] : batch[-1] + 1]
                inputs = outputs.__class__(**inputs)
            else:
                inputs = tuple()
                for base_output in outputs:
                    inputs = inputs + (base_output[batch],)
            if cls_out is not None:
                cls_input = cls_out[batch]
            else:
                cls_input = None
            return inputs, cls_input

        # Pass invertible adapter if we have one
        if hasattr(self.base_model, "get_invertible_adapter"):
            inv_adapter = self.base_model.get_invertible_adapter()
            if inv_adapter:
                kwargs["invertible_adapter"] = inv_adapter

        for head in used_heads:
            if head not in self.heads:
                raise ValueError("Unknown head_name '{}'".format(head))
        if isinstance(self.active_head, BatchSplit):
            if sum(self.active_head.batch_sizes) != all_outputs[0].size()[0]:
                raise ValueError(
                    "The specified batch sizes {} do not match the actual batch size {}".format(
                        self.active_head.batch_sizes, all_outputs[0].size()[0]
                    )
                )
            head_outputs = []
            labels = kwargs.pop("labels", None)
            for i, head in enumerate(self.active_head):
                head_module = self.heads[head]
                batch_idx = range(sum(self.active_head.batch_sizes[:i]), sum(self.active_head.batch_sizes[: i + 1]))
                kwargs["labels"] = labels[batch_idx] if labels is not None else None
                head_inputs, head_cls_input = _get_head_input(all_outputs, cls_output, batch_idx)
                # head_attention = attention_mask[batch_idx] if attention_mask is not None else None
                head_output = head_module(head_inputs, head_cls_input, attention_mask, return_dict, **kwargs)
                head_outputs.append(head_output)
            combined_loss = (
                sum([out["loss"] for out in head_outputs])
                if all("loss" in out and out["loss"] is not None for out in head_outputs)
                else None
            )
            return_output = MultiHeadOutput(head_outputs=head_outputs, loss=combined_loss)
        elif self.has_parallel_adapters or isinstance(self.active_head, Parallel):
            if len(self.active_head) != self.config.adapters.active_setup.parallel_channels:
                raise ValueError("The number of parallel adapters and the number of active heads must match.")
            orig_batch_size = all_outputs[0].shape[0] // self.config.adapters.active_setup.parallel_channels
            head_outputs = []
            for i, head in enumerate(self.active_head):
                head_module = self.heads[head]
                batch_idx = range(i * orig_batch_size, (i + 1) * orig_batch_size)
                head_inputs, head_cls_input = _get_head_input(all_outputs, cls_output, batch_idx)
                head_output = head_module(head_inputs, head_cls_input, attention_mask, return_dict, **kwargs)
                head_outputs.append(head_output)
            combined_loss = (
                torch.sum(torch.stack([out["loss"] for out in head_outputs]))
                if all("loss" in out and out["loss"] is not None for out in head_outputs)
                else None
            )
            return_output = MultiHeadOutput(head_outputs=head_outputs, loss=combined_loss)
        elif len(used_heads) > 1:
            head_outputs = []
            for head in used_heads:
                head_module = self.heads[head]
                head_outputs.append(head_module(all_outputs, cls_output, attention_mask, return_dict, **kwargs))
            return_output = MultiHeadOutput(head_outputs=head_outputs)
        else:
            head_module = self.heads[used_heads[0]]
            return_output = head_module(all_outputs, cls_output, attention_mask, return_dict, **kwargs)

        if isinstance(return_output, ModelOutput):
            for attr in ForwardContext.context_attributes:
                if attr not in return_output and attr in all_outputs:
                    return_output[attr] = all_outputs[attr]
        return return_output

    def get_labels_dict(self, head_name=None):
        """
        Returns the id2label dict for the given hea

        Args:
            head_name: (str, optional) the name of the head which labels should be returned. Default is None.
            If the name is None the labels of the active head are returned

        Returns: id2label

        """
        if head_name is None:
            head_name = self.active_head
        if head_name is None:
            raise ValueError("No head name given and no active head in the model")
        if "label2id" in self.heads[head_name].config.keys() and self.heads[head_name].config["label2id"] is not None:
            return {id_: label for label, id_ in self.heads[head_name].config["label2id"].items()}
        else:
            return None

    def get_labels(self, head_name=None):
        """
        Returns the labels the given head is assigning/predictin

        Args:
            head_name: (str, optional) the name of the head which labels should be returned. Default is None.
            If the name is None the labels of the active head are returned

        Returns: labels

        """
        label_dict = self.get_labels_dict(head_name)
        if label_dict is None:
            return None
        else:
            return list(label_dict.values())
