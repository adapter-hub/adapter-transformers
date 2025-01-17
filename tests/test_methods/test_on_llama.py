from transformers.models.llama.configuration_llama import LlamaConfig

from .generator import *


class LlamaAdapterTestBase(TextAdapterTestBase):
    config_class = LlamaConfig
    config = make_config(
        LlamaConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        pad_token_id=0,
    )
    tokenizer_name = "openlm-research/open_llama_13b"


method_tests = generate_method_tests(LlamaAdapterTestBase, excluded_tests=["PromptTuning"])

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class


@require_torch
class ClassConversion(
    ModelClassConversionTestMixin,
    LlamaAdapterTestBase,
    unittest.TestCase,
):
    def test_conversion_question_answering_model(self):
        raise self.skipTest("We don't support the Llama QA model.")
