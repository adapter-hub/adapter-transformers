import logging
from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .adapter_composition import AdapterCompositionBlock
from .adapter_model_mixin import ModelWithHeadsAdaptersMixin
from .adapter_modeling import Activation_Function_Class
from .modeling_outputs import (
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


logger = logging.getLogger(__name__)


# Let this class inherit from nn.Sequential to provide iterable access as before
class PredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config
        pred_head = []
        for l in range(self.config["layers"]):
            pred_head.append(nn.Dropout(model_config.hidden_dropout_prob))
            if l < self.config["layers"] - 1:
                pred_head.append(nn.Linear(model_config.hidden_size, model_config.hidden_size))
                pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            else:
                if "num_labels" in self.config:
                    pred_head.append(nn.Linear(model_config.hidden_size, self.config["num_labels"]))
                else:  # used for multiple_choice head
                    pred_head.append(nn.Linear(model_config.hidden_size, 1))
        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent


class ClassificationHead(PredictionHead):
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        cls_output = cls_output if cls_output is not None else outputs[0][:, 0]
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
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "multilabel_classification",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        cls_output = cls_output if cls_output is not None else outputs[0][:, 0]
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
    def __init__(self, head_name, num_choices, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "multiple_choice",
            "num_choices": num_choices,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
        }
        self.build(model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=None, **kwargs):
        cls_output = cls_output if cls_output is not None else outputs[0][:, 0]
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
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "tagging",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
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
    def __init__(self, head_name, num_labels, layers, activation_function, id2label, model):
        super().__init__(head_name)
        self.config = {
            "head_type": "question_answering",
            "num_labels": num_labels,
            "layers": layers,
            "activation_function": activation_function,
            "label2id": {label: id_ for id_, label in id2label.items()} if id2label else None,
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


class ModelWithFlexibleHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin, ABC):
    """Adds flexible prediction heads to a model class. Implemented by the XModelWithHeads classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.config, "custom_heads"):
            self.config.custom_heads = {}
        self.active_head = None

    def _init_head_modules(self):
        # this dict is _only_ used for saving & reloading the configs and should not be modified otherwise
        if not hasattr(self.config, "prediction_heads"):
            self.config.prediction_heads = {}
        self.heads = nn.ModuleDict(dict())
        # add modules for heads in config
        for head_name, config in self.config.prediction_heads.items():
            self.add_prediction_head_from_config(head_name, config)

    @abstractmethod
    def add_prediction_head_from_config(self, head_name, config, overwrite_ok=False):
        pass

    def get_prediction_heads_config(self):
        heads = {}
        for head_name, head in self.heads.items():
            heads[head_name] = head.config
        return heads

    def register_custom_head(self, identifier, head):
        self.config.custom_heads[identifier] = head

    @property
    def active_head(self):
        return self._active_head

    @active_head.setter
    def active_head(self, head_name):
        self._active_head = head_name
        if head_name is not None and head_name in self.heads:
            self.config.label2id = self.heads[head_name].config["label2id"]
            self.config.id2label = self.get_labels_dict(head_name)

    def set_active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        """
        Sets the adapter modules to be used by default in every forward pass. This setting can be overriden by passing
        the `adapter_names` parameter in the `foward()` pass. If no adapter with the given name is found, no module of
        the respective type will be activated. In case the calling model class supports named prediction heads, this
        method will attempt to activate a prediction head with the name of the last adapter in the list of passed
        adapter names.

        Args:
            adapter_setup (list): The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        self.base_model.set_active_adapters(adapter_setup)
        # use last adapter name as name of prediction head
        if self.active_adapters:
            head_name = self.active_adapters.last()
            if head_name in self.heads:
                self.active_head = head_name

            else:
                logger.info("No prediction head for task_name '{}' available.".format(head_name))

    def add_custom_head(self, head_name, config, overwrite_ok=False):
        if config["head_type"] in self.config.custom_heads:
            head = self.config.custom_heads[config["head_type"]](head_name, config, self)
            self.add_prediction_head(head, overwrite_ok)
        else:
            raise AttributeError(
                "The given head as a head_type that is not registered as a custom head yet."
                " Please register the head first."
            )

    def add_prediction_head(
        self,
        head: PredictionHead,
        overwrite_ok: bool = False,
    ):

        if head.name not in self.heads or overwrite_ok:
            self.heads[head.name] = head
            # add reference to model config to save all head configs too
            self.config.prediction_heads[head.name] = head.config

            if "label2id" not in head.config.keys() or head.config["label2id"] is None:
                if "num_labels" in head.config.keys():
                    head.config["label2id"] = {"LABEL_" + str(num): num for num in range(head.config["num_labels"])}
                if "num_choices" in head.config.keys():
                    head.config["label2id"] = {"LABEL_" + str(num): num for num in range(head.config["num_choices"])}

            logger.info(f"Adding head '{head.name}' with config {head.config}.")
            #             self._add_prediction_head_module(head.name)
            self.active_head = head.name

        else:
            raise ValueError(
                f"Model already contains a head with name '{head.name}'. Use overwrite_ok=True to force overwrite."
            )

    def forward_head(
        self, all_outputs, head_name=None, cls_output=None, attention_mask=None, return_dict=False, **kwargs
    ):

        head_name = head_name or self.active_head
        if not head_name:
            logger.debug("No prediction head is used.")
            return all_outputs

        if head_name not in self.heads:
            raise ValueError("Unknown head_name '{}'".format(head_name))
        head = self.heads[head_name]

        return head(all_outputs, cls_output, attention_mask, return_dict, **kwargs)

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
        if "label2id" in self.heads[head_name].config.keys():
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
