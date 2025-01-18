import copy
import os
import random
import tempfile
import unittest

import torch

from adapters import BertAdapterModel, Fuse
from adapters.utils import SAFE_ADAPTERFUSION_WEIGHTS_NAME, SAFE_HEAD_WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from transformers import BertConfig
from transformers.testing_utils import torch_device


class SafetensorsTest(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
        )

    def get_input_samples(self, shape, vocab_size=5000):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()
        in_data = {"input_ids": input_ids}

        return in_data

    def test_safetensors_adapter(self):
        model1 = BertAdapterModel(self.config)
        model2 = copy.deepcopy(model1)
        model1.eval()
        model2.eval()

        name = "test_adapter"
        model1.add_adapter(name)
        model1.add_classification_head(name, num_labels=2)
        model1.set_active_adapters(name)
        temp_dir = tempfile.TemporaryDirectory()

        # Save & reload adapter
        model1.save_adapter(temp_dir.name, name, use_safetensors=True)
        # Check that there are actually weights saved
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, SAFE_WEIGHTS_NAME)))
        # also tests that set_active works
        loading_info = {}
        model2.load_adapter(temp_dir.name, loading_info=loading_info, use_safetensors=True)
        # check if all weights were loaded
        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))
        # check if adapter was correctly loaded
        self.assertTrue(name in model2.adapters_config)
        model2.set_active_adapters(name)

        # check equal output
        input_data = self.get_input_samples((2, 32))
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

        try:
            temp_dir.cleanup()
        except Exception:
            pass

    def test_safetensors_head(self):
        model1 = BertAdapterModel(self.config)
        model2 = copy.deepcopy(model1)
        model1.eval()
        model2.eval()

        name = "test_adapter"
        model1.add_classification_head(name, num_labels=2)
        model1.active_head = name
        temp_dir = tempfile.TemporaryDirectory()

        # Save & reload head
        model1.save_head(temp_dir.name, name, use_safetensors=True)
        # Check that there are actually weights saved
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, SAFE_HEAD_WEIGHTS_NAME)))
        model2.load_head(temp_dir.name, use_safetensors=True)
        # check if adapter was correctly loaded
        self.assertTrue(name in model2.heads)
        model2.active_head = name

        # check equal output
        input_data = self.get_input_samples((2, 32))
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

        try:
            temp_dir.cleanup()
        except Exception:
            pass

    def test_safetensors_adapter_fusion(self):
        model1 = BertAdapterModel(self.config)
        adapter_names = ["test_adapter1", "test_adapter2"]
        for adapter_name in adapter_names:
            model1.add_adapter(adapter_name)
        model2 = copy.deepcopy(model1)
        model1.eval()
        model2.eval()

        fusion = Fuse(*adapter_names)
        model1.add_adapter_fusion(fusion)
        model1.set_active_adapters(fusion)
        temp_dir = tempfile.TemporaryDirectory()

        # Save & reload adapter
        model1.save_adapter_fusion(temp_dir.name, fusion, use_safetensors=True)
        # Check that there are actually weights saved
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, SAFE_ADAPTERFUSION_WEIGHTS_NAME)))
        # also tests that set_active works
        model2.load_adapter_fusion(temp_dir.name, use_safetensors=True)
        model2.set_active_adapters(fusion)

        # check equal output
        input_data = self.get_input_samples((2, 32))
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

        try:
            temp_dir.cleanup()
        except Exception:
            pass
