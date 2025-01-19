from transformers import RobertaConfig

from .generator import *


class RobertaAdapterTestBase(TextAdapterTestBase):
    config_class = RobertaConfig
    config = make_config(
        RobertaConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=50265,
    )
    tokenizer_name = "roberta-base"


method_tests = generate_method_tests(RobertaAdapterTestBase)

for test_name, test_class in method_tests.items():
    globals()[test_name] = test_class
