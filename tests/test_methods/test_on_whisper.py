from transformers import WhisperConfig

from .generator import *


class WhisperAdapterTestBase(AudioAdapterTestBase):
    config_class = WhisperConfig
    config = make_config(
        WhisperConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=51865,
    )
    tokenizer_name = "openai/whisper-small"
    sampling_rate = 16000
    decoder_start_token_id = 50257

    def test_parallel_training_lora(self):
        self.skipTest("Not supported for Whisper")


method_tests = generate_method_tests(WhisperAdapterTestBase, excluded_tests=["PromptTuning"])
for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
