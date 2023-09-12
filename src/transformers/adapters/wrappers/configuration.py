import copy

from ...configuration_utils import PretrainedConfig
from ...models.clip.configuration_clip import CLIPConfig
from ...models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
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
    "hubert": {},
    "mbart": {
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
    "xlm_roberta": {},
}
SUBMODEL_NAMES = {"clip": ["vision_config", "text_config"], "encoder-decoder": ["encoder", "decoder"]}


def wrap_config(config: PretrainedConfig) -> PretrainedConfig:
    """
    Makes required changes to a model config class to allow usage with adapters.

    Args:
        config (PretrainedConfig): The config to be wrapped.

    Returns:
        PretrainedConfig: The same config object, with modifications applied.
    """
    if getattr(config, "is_adaptable", False):
        return config

    # Init ModelAdaptersConfig
    if not hasattr(config, "adapters"):
        config.adapters = ModelAdaptersConfig()
    elif config.adapters is not None and not isinstance(config.adapters, ModelAdaptersConfig):
        config.adapters = ModelAdaptersConfig(**config.adapters)

    # Convert AdapterFusions from old format for backwards compatibility
    fusion_models = getattr(config, "adapter_fusion_models", [])
    fusion_config = getattr(config, "adapter_fusion", None)
    for fusion_adapter_names in fusion_models:
        config.adapters.add_fusion(fusion_adapter_names, config=fusion_config)

    # Make sure each class has its own attribute_map
    type(config).attribute_map = copy.deepcopy(type(config).attribute_map)
    # Ensure missing keys are in class
    if config.model_type in CONFIG_CLASS_KEYS_MAPPING:
        for key, value in CONFIG_CLASS_KEYS_MAPPING[config.model_type].items():
            if key not in config.attribute_map:
                config.attribute_map[key] = value

    # Ensure custom_heads attribute is present
    if not hasattr(config, "custom_heads"):
        config.custom_heads = {}

    if isinstance(config, EncoderDecoderConfig):
        # make sure adapter config is shared
        wrap_config(config.encoder)
        wrap_config(config.decoder)
        config.decoder.adapters = config.encoder.adapters
        config.adapters = config.encoder.adapters
    elif isinstance(config, CLIPConfig):
        wrap_config(config.vision_config)
        wrap_config(config.text_config)
        config.vision_config.adapters = config.adapters
        config.text_config.adapters = config.adapters
    config.is_adaptable = True

    return config
