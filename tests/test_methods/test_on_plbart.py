from transformers import PLBartConfig

from .generator import *


class PLBartAdapterTestBase(TextAdapterTestBase):
    config_class = PLBartConfig
    config = make_config(
        PLBartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        scale_embedding=False,  # Required for embedding tests
    )
    tokenizer_name = "uclanlp/plbart-base"


method_tests = generate_method_tests(PLBartAdapterTestBase, excluded_tests=["PromptTuning"])

for test_name, test_class in method_tests.items():
    globals()[test_name] = test_class
