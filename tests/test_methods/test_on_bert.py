from transformers import BertConfig

from .generator import TextAdapterTestBase, generate_method_tests, make_config


class BertAdapterTestBase(TextAdapterTestBase):
    config_class = BertConfig
    config = make_config(
        BertConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "bert-base-uncased"


method_tests = generate_method_tests(BertAdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
