from transformers.models.mistral.configuration_mistral import MistralConfig

from .generator import *


class MistralAdapterTestBase(TextAdapterTestBase):
    config_class = MistralConfig
    config = make_config(
        MistralConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=8,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        pad_token_id=0,
    )
    tokenizer_name = "HuggingFaceH4/zephyr-7b-beta"


test_methods = generate_method_tests(MistralAdapterTestBase, excluded_tests=["PromptTuning", "ConfigUnion"])

for test_class_name, test_class in test_methods.items():
    globals()[test_class_name] = test_class
