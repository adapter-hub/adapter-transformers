import json
import unittest
from dataclasses import FrozenInstanceError, dataclass

from adapters import (
    ADAPTER_CONFIG_MAP,
    AdapterConfig,
    ConfigUnion,
    DoubleSeqBnConfig,
    LoRAConfig,
    MAMConfig,
    ParBnConfig,
    PrefixTuningConfig,
    SeqBnConfig,
)
from transformers.testing_utils import require_torch


@require_torch
class AdapterConfigTest(unittest.TestCase):
    def test_config_immutable(self):
        def set_attr(config: AdapterConfig):
            config.non_linearity = "dummy"
            config.r = -1  # for LoRA
            config.prompt_length = -1  # for PromptTuning

        for config in ADAPTER_CONFIG_MAP.values():
            if isinstance(config, ConfigUnion):
                continue
            with self.subTest(config=config.__class__.__name__):
                self.assertRaises(FrozenInstanceError, lambda: set_attr(config))

    def test_custom_attr(self):
        for config in ADAPTER_CONFIG_MAP.values():
            with self.subTest(config=config.__class__.__name__):
                # create a copy to leave original untouched
                config = config.replace()
                config.dummy_attr = "test_value"
                self.assertEqual(config.dummy_attr, "test_value")

    def test_custom_class(self):
        @dataclass
        class CustomAdapterConfig(SeqBnConfig):
            custom_attr: str = "test_value"

        config = CustomAdapterConfig()
        config_dict = config.to_dict()
        self.assertEqual(config_dict["custom_attr"], "test_value")
        # When calling load on an AdapterConfig instance, don't change the class of the config.
        config = AdapterConfig.load(config, custom_attr="test_value_2")
        self.assertTrue(isinstance(config, CustomAdapterConfig))
        self.assertEqual(config["custom_attr"], "test_value_2")

    def test_config_union_valid(self):
        unions = [
            [PrefixTuningConfig(), ParBnConfig()],
            [PrefixTuningConfig(), SeqBnConfig()],
            [DoubleSeqBnConfig(mh_adapter=False), DoubleSeqBnConfig(output_adapter=False, reduction_factor=2)],
            [SeqBnConfig(leave_out=[9, 10, 11], reduction_factor=2), SeqBnConfig(leave_out=list(range(9)))],
        ]
        for union in unions:
            with self.subTest(union=union):
                config = ConfigUnion(*union)
                self.assertEqual(config.architecture, "union")

                # make sure serialization/ deserialization works
                config_dict = config.to_dict()
                config_dict = json.loads(json.dumps(config_dict))
                config_new = ConfigUnion.from_dict(config_dict)
                self.assertEqual(config, config_new)

                self.assertIsInstance(config_new[0], AdapterConfig)
                self.assertIsInstance(config_new[1], AdapterConfig)

    def test_config_union_invalid(self):
        unions = [
            ([MAMConfig(), SeqBnConfig()], TypeError),
            ([SeqBnConfig(), SeqBnConfig()], ValueError),
            ([SeqBnConfig(), DoubleSeqBnConfig()], ValueError),
        ]
        for union, error_type in unions:
            with self.subTest(union=union):
                self.assertRaises(error_type, ConfigUnion, *union)

    def test_config_string_valid(self):
        to_test = [
            ("double_seq_bn", DoubleSeqBnConfig()),
            ("seq_bn[reduction_factor=2, leave_out=[11]]", SeqBnConfig(reduction_factor=2, leave_out=[11])),
            (
                "par_bn[reduction_factor={'0': 8, '1': 8, 'default': 16}]",
                ParBnConfig(reduction_factor={"0": 8, "1": 8, "default": 16}),
            ),
            ("prefix_tuning[prefix_length=30, flat=True]", PrefixTuningConfig(prefix_length=30, flat=True)),
            ("lora[r=200,alpha=8]", LoRAConfig(r=200, alpha=8)),
            ("prefix_tuning|par_bn", ConfigUnion(PrefixTuningConfig(), ParBnConfig())),
            ("lora[attn_matrices=['k', 'v']]", LoRAConfig(attn_matrices=["k", "v"])),
            (
                "lora[use_gating=True]|prefix_tuning[use_gating=True]|seq_bn[use_gating=True]",
                ConfigUnion(
                    LoRAConfig(use_gating=True), PrefixTuningConfig(use_gating=True), SeqBnConfig(use_gating=True)
                ),
            ),
        ]
        for config_str, config in to_test:
            with self.subTest(config_str=config_str):
                config_new = AdapterConfig.load(config_str)
                self.assertEqual(config, config_new)

    def test_config_string_invalid(self):
        to_test = [
            ("seq_bn[invalid_key=2]", TypeError),
            ("lora[r=8]|invalid_name", ValueError),
            ("prefix_tuning[flat=True", ValueError),
            ("double_seq_bn[reduction_factor=dict(default=1)]", ValueError),
        ]
        for config_str, error_type in to_test:
            with self.subTest(config_str=config_str):
                self.assertRaises(error_type, AdapterConfig.load, config_str)
