from transformers import XmodConfig

from .generator import *


class XmodAdapterTestBase(TextAdapterTestBase):
    config_class = XmodConfig
    config = make_config(
        XmodConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        vocab_size=250002,
        max_position_embeddings=512,
        default_language="en_XX",
    )
    tokenizer_name = "xlm-roberta-base"


method_tests = generate_method_tests(XmodAdapterTestBase, not_supported=["ConfigUnion", "Embeddings"])
for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
