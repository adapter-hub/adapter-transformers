from transformers import DebertaConfig

from .generator import *


class DebertaAdapterTestBase(TextAdapterTestBase):
    config_class = DebertaConfig
    config = make_config(
        DebertaConfig,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        relative_attention=True,
        pos_att_type="p2c|c2p",
    )
    tokenizer_name = "microsoft/deberta-base"

    def test_parallel_training_lora(self):
        self.skipTest("Not supported for DeBERTa")


method_tests = generate_method_tests(DebertaAdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
