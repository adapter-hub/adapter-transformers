from transformers import GPT2Config

from .generator import *


class GPT2AdapterTestBase(TextAdapterTestBase):
    config_class = GPT2Config
    config = make_config(
        GPT2Config,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=50256,
    )
    tokenizer_name = "gpt2"


method_tests = generate_method_tests(GPT2AdapterTestBase, excluded_tests=["PromptTuning"])

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class


@require_torch
@pytest.mark.composition
class Composition(
    GPT2AdapterTestBase,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    unittest.TestCase,
):
    def test_parallel_training_lora(self):
        self.skipTest("Not supported for GPT2")
