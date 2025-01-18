from transformers import ElectraConfig

from .generator import *


class ElectraAdapterTestBase(TextAdapterTestBase):
    config_class = ElectraConfig
    config = make_config(
        ElectraConfig,
        # vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "google/electra-base-generator"


method_tests = generate_method_tests(ElectraAdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
