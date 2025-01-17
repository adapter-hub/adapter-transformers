import copy
import logging
import re


logger = logging.getLogger(__name__)

# The "layers" attributes in the configs below map from static head module names to flex head module names.
# In this context, "None" refers to a flex-head layer without weights (e.g. dropout, acts).
STATIC_TO_FLEX_HEAD_MAP = {
    # ALBERT
    "AlbertForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "AlbertForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "AlbertForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "AlbertForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "AlbertForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu_new",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "predictions.dense",
            None,
            "predictions.LayerNorm",
            "predictions.decoder",
        ],
    },
    # BEIT
    "BeitForImageClassification": {
        "config": {
            "head_type": "image_classification",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    # BERT
    "BertForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "BertForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "BertForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "BertForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "BertForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "cls.predictions.transform.dense",
            None,
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ],
    },
    "BertLMHeadModel": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "cls.predictions.transform.dense",
            None,
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ],
    },
    # BertGeneration
    "BertGenerationDecoder": {
        "config": {
            "head_type": "causal_lm",
            "layers": 1,
            "activation_function": None,
            "bias": True,
        },
        "layers": [
            "lm_head.decoder",
        ],
    },
    # RoBERTa
    "RobertaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
            "use_pooler": False,
        },
        "layers": [None, "classifier.dense", None, None, "classifier.out_proj"],
    },
    "RobertaForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "RobertaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "RobertaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "RobertaForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    "RobertaForCausalLM": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    # XLM-RoBERTa
    "XLMRobertaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
            "use_pooler": False,
        },
        "layers": [None, "classifier.dense", None, None, "classifier.out_proj"],
    },
    "XLMRobertaForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "XLMRobertaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "XLMRobertaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "XLMRobertaForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    "XLMRobertaForCausalLM": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    # Xmod
    "XmodForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
            "use_pooler": False,
        },
        "layers": [None, "classifier.dense", None, None, "classifier.out_proj"],
    },
    "XmodForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "XmodForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "XmodForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "XmodForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    "XmodForCausalLM": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    # BART
    "BartForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": [
            None,
            "classification_head.dense",
            None,
            None,
            "classification_head.out_proj",
        ],
    },
    "BartForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "BartForConditionalGeneration": {
        "config": {
            "head_type": "seq2seq_lm",
        },
        "layers": ["lm_head"],
    },
    # MBART
    "MBartForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": [
            None,
            "classification_head.dense",
            None,
            None,
            "classification_head.out_proj",
        ],
    },
    "MBartForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "MBartForConditionalGeneration": {
        "config": {
            "head_type": "seq2seq_lm",
        },
        "layers": ["lm_head"],
    },
    # PLBART
    "PLBartForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": [
            None,
            "classification_head.dense",
            None,
            None,
            "classification_head.out_proj",
        ],
    },
    "PLBartForConditionalGeneration": {
        "config": {
            "head_type": "seq2seq_lm",
        },
        "layers": ["lm_head"],
    },
    # MT5
    "MT5ForConditionalGeneration": {
        "config": {
            "head_type": "seq2seq_lm",
        },
        "layers": ["lm_head"],
    },
    "MT5ForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "activation_function": None,
            "layers": 1,
        },
        "layers": [None, "qa_outputs"],
    },
    "MT5ForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": [
            None,
            "classification_head.dense",
            None,
            None,
            "classification_head.out_proj",
        ],
    },
    # DistilBERT
    "DistilBertForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "relu",
        },
        "layers": [None, "pre_classifier", None, None, "classifier"],
    },
    "DistilBertForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 2,
            "activation_function": "relu",
        },
        "layers": [None, "pre_classifier", None, None, "classifier"],
    },
    "DistilBertForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "DistilBertForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "DistilBertForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["vocab_transform", None, "vocab_layer_norm", "vocab_projector"],
    },
    # GPT-2
    "GPT2ForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "bias": False,
        },
        "layers": [None, "score"],
    },
    "GPT2LMHeadModel": {
        "config": {
            "head_type": "causal_lm",
        },
        "layers": ["lm_head"],
    },
    "GPT2ForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "GPT2ForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    # GPT-J
    "GPTJForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "bias": False,
        },
        "layers": [None, "score"],
    },
    "GPTJForCausalLM": {
        "config": {
            "head_type": "causal_lm",
            "bias": True,
        },
        "layers": ["lm_head"],
    },
    "GPTJForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    # T5
    "T5ForConditionalGeneration": {
        "config": {
            "head_type": "seq2seq_lm",
        },
        "layers": ["lm_head"],
    },
    "T5ForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "activation_function": None,
            "layers": 1,
        },
        "layers": [None, "qa_outputs"],
    },
    "T5ForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
        },
        "layers": [
            None,
            "classification_head.dense",
            None,
            None,
            "classification_head.out_proj",
        ],
    },
    # DeBERTaV2
    "DebertaV2ForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "gelu",
            "use_pooler": False,
        },
        "layers": [None, "pooler.dense", None, None, "classifier"],
    },
    "DebertaV2ForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["dropout", "classifier"],
    },
    "DebertaV2ForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "DebertaV2ForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "cls.predictions.transform.dense",
            None,
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ],
    },
    "DebertaV2ForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 2,
            "activation_function": "gelu",
            "use_pooler": False,
        },
        "layers": [None, "pooler.dense", None, None, "classifier"],
    },
    # DeBERTa
    "DebertaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "gelu",
            "use_pooler": False,
        },
        "layers": [None, "pooler.dense", None, None, "classifier"],
    },
    "DebertaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": ["dropout", "classifier"],
    },
    "DebertaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "DebertaForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "cls.predictions.transform.dense",
            None,
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ],
    },
    # ViT
    "ViTForImageClassification": {
        "config": {
            "head_type": "image_classification",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    # Llama
    "LlamaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "dropout_prob": 0,
            "activation_function": None,
            "bias": False,
        },
        "layers": [None, "score"],
    },
    "LlamaForCausalLM": {
        "config": {
            "head_type": "causal_lm",
        },
        "layers": ["lm_head"],
    },
    "LlamaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "LlamaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "score"],
    },
    # Mistral
    "MistralForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "dropout_prob": 0,
            "activation_function": None,
            "bias": False,
        },
        "layers": [None, "score"],
    },
    "MistralForCausalLM": {
        "config": {
            "head_type": "causal_lm",
        },
        "layers": ["lm_head"],
    },
    "MistralForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "score"],
    },
    "MistralForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    # Electra
    "ElectraForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "ElectraForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "gelu",
            "bias": True,
        },
        "layers": [None, "classifier.dense", None, None, "classifier.out_proj"],
    },
    "ElectraForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "ElectraForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 2,
            "activation_function": "gelu",
            "use_pooler": False,
        },
        "layers": [None, "sequence_summary.summary", None, None, "classifier"],
    },
    "ElectraForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "generator_predictions.dense",
            None,
            "generator_predictions.LayerNorm",
            "generator_lm_head",
        ],
    },
    "ElectraForCausalLM": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "generator_predictions.dense",
            None,
            "generator_predictions.LayerNorm",
            "generator_lm_head",
        ],
    },
    "WhisperForConditionalGeneration": {
        "config": {
            "head_type": "seq2seq_lm",
            "layers": 1,
            "activation_function": None,
            "layer_norm": False,
            "bias": False,
        },
        "layers": ["proj_out"],
    },
    "MllamaForConditionalGeneration": {
        "config": {
            "head_type": "causal_lm",
            "layers": 1,
            "activation_function": None,
            "layer_norm": False,
            "bias": False,
        },
        "layers": ["language_model.lm_head"],
    },
}


def _regex_list_rename_func(k, rename_list):
    for o, n in rename_list:
        match = re.match(o, k)
        if match:
            return n.format(match.group(1))
    return k


def get_head_config_and_rename_list(model_class_name, head_name, label2id, num_labels=None):
    if label2id is None:
        logger.warning(
            "No valid map of labels in label2id. Falling back to default (num_labels=2). This may cause errors during"
            " loading!"
        )
        label2id = {"LABEL_" + str(i): i for i in range(2)}
    # num_labels is optional (e.g. for regression, when no map given)
    num_labels = num_labels or len(label2id)
    data = STATIC_TO_FLEX_HEAD_MAP[model_class_name]
    # copy config to keep original mapping untouched
    config = copy.deepcopy(data["config"])
    if config["head_type"] == "multiple_choice":
        config["num_choices"] = num_labels
        config["label2id"] = label2id
    elif config["head_type"] not in ["causal_lm", "masked_lm", "seq2seq_lm"]:
        config["num_labels"] = num_labels
        config["label2id"] = label2id
    # rename
    rename_list = []
    i = 0
    for name in data["layers"]:
        if name is not None:
            escaped_name = re.escape(name)
            rename_list.append((rf"{escaped_name}\.(\S+)", f"heads.{head_name}.{i}.{{0}}"))
        i += 1
    rename_func = lambda k, rename_list=rename_list: _regex_list_rename_func(k, rename_list)

    return config, rename_func
