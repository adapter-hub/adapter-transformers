from transformers import BartConfig

from .generator import *


class BartAdapterTestBase(TextAdapterTestBase):
    config_class = BartConfig
    config = make_config(
        BartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
    )
    tokenizer_name = "facebook/bart-base"


method_tests = generate_method_tests(BartAdapterTestBase, excluded_tests=["PromptTuning"])

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
