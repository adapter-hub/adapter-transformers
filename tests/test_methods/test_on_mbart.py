from transformers import MBartConfig

from .generator import *


class MBartAdapterTestBase(TextAdapterTestBase):
    config_class = MBartConfig
    config = make_config(
        MBartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=250027,
    )
    tokenizer_name = "facebook/mbart-large-cc25"

    def test_parallel_training_lora(self):
        self.skipTest("Not supported for MBart")


method_tests = generate_method_tests(
    MBartAdapterTestBase, redundant=["ConfigUnion", "Embeddings"], not_supported=["PromptTuning"]
)
for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
