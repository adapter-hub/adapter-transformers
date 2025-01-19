from transformers import DebertaV2Config

from .generator import *


class DebertaV2AdapterTestBase(TextAdapterTestBase):
    config_class = DebertaV2Config
    config = make_config(
        DebertaV2Config,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        relative_attention=True,
        pos_att_type="p2c|c2p",
    )
    tokenizer_name = "microsoft/deberta-v3-base"


method_tests = generate_method_tests(DebertaV2AdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
