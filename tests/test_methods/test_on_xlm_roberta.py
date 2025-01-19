from transformers import XLMRobertaConfig

from .generator import *


class XLMRobertaAdapterTestBase(TextAdapterTestBase):
    config_class = XLMRobertaConfig
    config = make_config(
        XLMRobertaConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=250002,
    )
    tokenizer_name = "xlm-roberta-base"


method_tests = generate_method_tests(XLMRobertaAdapterTestBase, redundant=["ConfigUnion", "Embeddings"])
for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
