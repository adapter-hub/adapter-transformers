import unittest
from dataclasses import FrozenInstanceError

from adapters import ADAPTERFUSION_CONFIG_MAP, AdapterFusionConfig
from transformers.testing_utils import require_torch


@require_torch
class AdapterFusionConfigTest(unittest.TestCase):

    config_names = ADAPTERFUSION_CONFIG_MAP.keys()

    def test_config_load(self):
        for config_name in self.config_names:
            with self.subTest(config_name=config_name):
                config = AdapterFusionConfig.load(config_name, temperature=True)
                self.assertTrue(isinstance(config, AdapterFusionConfig))
                self.assertEqual(config.temperature, True)

    def test_config_immutable(self):
        def set_attr(config: AdapterFusionConfig):
            config.temperature = True

        for config in ADAPTERFUSION_CONFIG_MAP.values():
            with self.subTest(config=config.__class__.__name__):
                self.assertRaises(FrozenInstanceError, lambda: set_attr(config))

    def test_custom_attr(self):
        for config in ADAPTERFUSION_CONFIG_MAP.values():
            with self.subTest(config=config.__class__.__name__):
                config.dummy_attr = "test_value"
                self.assertEqual(config.dummy_attr, "test_value")
