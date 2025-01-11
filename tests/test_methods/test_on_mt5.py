from transformers import MT5Config

from .generator import *


@require_torch
class MT5AdapterTestBase(TextAdapterTestBase):
    config_class = MT5Config
    config = make_config(
        MT5Config,
        d_model=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=4,
        d_kv=16 // 4,
        tie_word_embeddings=False,
        decoder_start_token_id=0,
    )
    tokenizer_name = "google/mt5-base"


method_tests = generate_method_tests(MT5AdapterTestBase, excluded_tests=["PromptTuning", "ConfigUnion"])

for test_name, test_class in method_tests.items():
    globals()[test_name] = test_class
