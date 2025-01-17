import copy
from typing import Optional

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from ..configuration import ModelAdaptersConfig


CONFIG_CLASS_KEYS_MAPPING = {
    "albert": {
        "classifier_dropout": "classifier_dropout_prob",
    },
    "bart": {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
    },
    "beit": {},
    "bert": {},
    "clip_vision_model": {
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
    },
    "clip_text_model": {
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
    },
    "distilbert": {
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
        "classifier_dropout": "seq_classif_dropout",
    },
    "gpt2": {
        "hidden_dropout_prob": "resid_pdrop",
        "attention_probs_dropout_prob": "attn_pdrop",
    },
    "gptj": {
        "hidden_dropout_prob": "resid_pdrop",
        "attention_probs_dropout_prob": "attn_pdrop",
    },
    "mbart": {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
    },
    "plbart": {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
    },
    "roberta": {},
    "t5": {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "hidden_dropout_prob": "dropout_rate",
        "attention_probs_dropout_prob": "dropout_rate",
    },
    "vit": {},
    "whisper": {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "hidden_dropout_prob": "dropout",
        "attention_probs_dropout_prob": "attention_dropout",
    },
    "xlm_roberta": {},
    # TODO: add mllama
}
SUBMODEL_NAMES = {"clip": ["vision_config", "text_config"], "encoder-decoder": ["encoder", "decoder"]}


def init_adapters_config(
    model: PreTrainedModel, model_config: PretrainedConfig, adapters_config: Optional[ModelAdaptersConfig] = None
):
    """Initializes the adapters config object of the model to enable adapter support. Also make required changes to the
    model's config.

        Args:
            model (PreTrainedModel): The model for which to add the adapters config.
            model_config (PretrainedConfig): The model's config.
            adapters_config (ModelAdaptersConfig): The adapters config to be added.
    """
    # Make sure config is wrapped
    model.config = model_config
    wrap_config(model.config)

    # Init ModelAdaptersConfig
    if adapters_config is not None:
        model.adapters_config = adapters_config
    elif not hasattr(model_config, "adapters"):
        model.adapters_config = ModelAdaptersConfig()
    elif model_config.adapters is not None and not isinstance(model_config.adapters, ModelAdaptersConfig):
        model.adapters_config = ModelAdaptersConfig(**model_config.adapters)

    # Convert AdapterFusions from old format for backwards compatibility
    fusion_models = getattr(model_config, "adapter_fusion_models", [])
    fusion_config = getattr(model_config, "adapter_fusion", None)
    for fusion_adapter_names in fusion_models:
        model.adapters_config.add_fusion(fusion_adapter_names, config=fusion_config)


def wrap_config(config: PretrainedConfig):
    """
    Makes required changes to a model config class to allow usage with adapters.

    Args:
        config (PretrainedConfig): The config to be wrapped.

    Returns:
        PretrainedConfig: The same config object, with modifications applied.
    """

    # Make sure each class has its own attribute_map
    type(config).attribute_map = copy.deepcopy(type(config).attribute_map)
    # Ensure missing keys are in class
    if config.model_type in CONFIG_CLASS_KEYS_MAPPING:
        for key, value in CONFIG_CLASS_KEYS_MAPPING[config.model_type].items():
            if key not in config.attribute_map:
                config.attribute_map[key] = value
