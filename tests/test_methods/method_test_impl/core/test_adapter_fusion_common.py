import copy
import os
import tempfile
from dataclasses import asdict

import torch

from adapters import ADAPTER_MODEL_MAPPING, ADAPTERFUSION_CONFIG_MAP, AdapterConfig, AutoAdapterModel, SeqBnConfig
from adapters.composition import Fuse
from adapters.utils import ADAPTERFUSION_WEIGHTS_NAME
from adapters.wrappers import load_model
from transformers.testing_utils import require_torch, torch_device


@require_torch
class AdapterFusionModelTestMixin:
    def test_add_adapter_fusion(self):
        config_name = "seq_bn"
        adapter_config = AdapterConfig.load(config_name)

        for adater_fusion_config_name, adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP.items():
            model = self.get_model()
            model.eval()

            with self.subTest(model_class=model.__class__.__name__, config=config_name):
                name1 = f"{config_name}-1"
                name2 = f"{config_name}-2"
                model.add_adapter(name1, config=config_name)
                model.add_adapter(name2, config=config_name)

                # adapter is correctly added to config
                self.assertTrue(name1 in model.adapters_config)
                self.assertTrue(name2 in model.adapters_config)
                self.assertEqual(asdict(adapter_config), asdict(model.adapters_config.get(name1)))
                self.assertEqual(asdict(adapter_config), asdict(model.adapters_config.get(name2)))

                model.add_adapter_fusion([name1, name2], adater_fusion_config_name)

                # check forward pass
                input_data = self.get_input_samples(config=model.config)
                model.set_active_adapters(Fuse(name1, name2))
                model.to(torch_device)
                adapter_output = model(**input_data)
                model.set_active_adapters(None)
                base_output = model(**input_data)
                self.assertEqual(len(adapter_output), len(base_output))
                self.assertFalse(torch.equal(adapter_output[0], base_output[0]))

    def test_add_adapter_fusion_different_config(self):
        model = self.get_model()
        model.eval()

        # fusion between a and b should be possible whereas fusion between a and c should fail
        model.add_adapter("a", config=SeqBnConfig(reduction_factor=16))
        model.add_adapter("b", config=SeqBnConfig(reduction_factor=2))
        model.add_adapter("c", config="double_seq_bn")

        # correct fusion
        model.add_adapter_fusion(["a", "b"])
        self.assertIn("a,b", model.adapters_config.fusions)
        # failing fusion
        self.assertRaises(ValueError, lambda: model.add_adapter_fusion(["a", "c"]))

    def test_delete_adapter_fusion(self):
        model = self.get_model()
        model.eval()

        name1 = "test_adapter_1"
        name2 = "test_adapter_2"
        model.add_adapter(name1, config="double_seq_bn")
        model.add_adapter(name2, config="double_seq_bn")
        self.assertTrue(name1 in model.adapters_config)
        self.assertTrue(name2 in model.adapters_config)

        model.add_adapter_fusion([name1, name2])
        self.assertTrue(",".join([name1, name2]) in model.adapters_config.fusions)

        model.delete_adapter_fusion([name1, name2])
        self.assertFalse(",".join([name1, name2]) in model.adapters_config.fusions)

    def test_load_adapter_fusion(self):
        for adater_fusion_config_name, adapter_fusion_config in ADAPTERFUSION_CONFIG_MAP.items():
            model1 = self.get_model()
            model1.eval()

            with self.subTest(model_class=model1.__class__.__name__):
                name1 = "name1"
                name2 = "name2"
                model1.add_adapter(name1)
                model1.add_adapter(name2)

                model2 = copy.deepcopy(model1)
                model2.eval()

                model1.add_adapter_fusion([name1, name2], adater_fusion_config_name)
                model1.set_active_adapters(Fuse(name1, name2))

                with tempfile.TemporaryDirectory() as temp_dir:
                    model1.save_adapter_fusion(temp_dir, ",".join([name1, name2]))
                    # also tests that set_active works
                    model2.load_adapter_fusion(temp_dir, set_active=True)
                # In another directory, also check that saving via passing a Fuse block works
                with tempfile.TemporaryDirectory() as temp_dir:
                    model1.save_adapter_fusion(temp_dir, Fuse(name1, name2))
                    self.assertTrue(os.path.exists(os.path.join(temp_dir, ADAPTERFUSION_WEIGHTS_NAME)))

                # check if adapter was correctly loaded
                self.assertEqual(model1.adapters_config.fusions.keys(), model2.adapters_config.fusions.keys())

                # check equal output
                in_data = self.get_input_samples(config=model1.config)
                model1.to(torch_device)
                model2.to(torch_device)
                output1 = model1(**in_data)
                output2 = model2(**in_data)
                self.assertEqual(len(output1), len(output2))
                self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_load_full_model_fusion(self):
        model1 = self.get_model()
        model1.eval()

        name1 = "name1"
        name2 = "name2"
        model1.add_adapter(name1)
        model1.add_adapter(name2)
        model1.add_adapter_fusion([name1, name2])
        # save & reload model
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            model1.save_pretrained(temp_dir)

            model2 = load_model(temp_dir, self.model_class)

        # check if AdapterFusion was correctly loaded
        self.assertTrue(model1.adapters_config.fusions == model2.adapters_config.fusions)

        # check equal output
        input_data = self.get_input_samples(config=model1.config)
        model1.set_active_adapters(Fuse(name1, name2))
        model2.set_active_adapters(Fuse(name1, name2))
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_model_config_serialization_fusion(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for k, v in ADAPTERFUSION_CONFIG_MAP.items():
            model = self.get_model()
            model.add_adapter("test1")
            model.add_adapter("test2")
            model.add_adapter_fusion(["test1", "test2"], config=v)
            # should not raise an exception
            model.config.to_json_string()

    def test_adapter_fusion_save_with_head(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model1 = AutoAdapterModel.from_config(self.config())
        model1.eval()

        name1 = "name1"
        name2 = "name2"
        head_name = "adapter_fusion_head"
        model1.add_adapter(name1)
        model1.add_adapter(name2)
        model2 = copy.deepcopy(model1)
        model2.eval()
        model1.add_adapter_fusion([name1, name2])
        self.add_head(model1, head_name)
        model1.set_active_adapters(Fuse(name1, name2))
        # save & reload model
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter_fusion(temp_dir, ",".join([name1, name2]), with_head=head_name)
            model2.load_adapter_fusion(temp_dir, set_active=True)

        self.assertTrue(head_name in model2.heads)
        self.assertEqual(model1.active_head, model2.active_head)
        self.assertEqual(model1.active_adapters, model2.active_adapters)

        # assert equal forward pass
        in_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**in_data)
        output2 = model2(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_output_adapter_fusion_attentions(self):
        model = self.get_model()
        model.eval()

        model.add_adapter("a")
        model.add_adapter("b")
        model.add_adapter_fusion(["a", "b"])
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        model.set_active_adapters(Fuse("a", "b"))
        output_1 = model(**input_data, output_adapter_fusion_attentions=True)

        self.assertEqual(len(output_1[0]), self.input_shape[0])
        self.assertTrue(hasattr(output_1, "adapter_fusion_attentions"))
        attention_scores = output_1.adapter_fusion_attentions["a,b"]
        self.assertEqual(len(list(model.iter_layers())), len(attention_scores))
        for k, per_layer_scores in attention_scores.items():
            self.assertEqual(len(per_layer_scores), 1)
            for k, v in per_layer_scores.items():
                self.assertEqual(self.input_shape[0], v.shape[0], k)

    def test_add_adapter_fusion_custom_name(self):
        config_name = "seq_bn"
        model = self.get_model()
        model.eval()

        name1 = f"{config_name}-1"
        name2 = f"{config_name}-2"
        model.add_adapter(name1, config=config_name)
        model.add_adapter(name2, config=config_name)

        # adapter is correctly added to config
        self.assertTrue(name1 in model.adapters_config)
        self.assertTrue(name2 in model.adapters_config)

        # add fusion with default name
        model.add_adapter_fusion([name1, name2])
        model.to(torch_device)

        # check forward pass
        input_data = self.get_input_samples(config=model.config)
        model.set_active_adapters(Fuse(name1, name2))
        fusion_default_ref_output = model(**input_data)

        # add fusion with custom name
        model.add_adapter_fusion([name1, name2], name="custom_name_fusion")
        model.to(torch_device)

        self.assertIn(f"{name1},{name2}", model.adapters_config.fusions)
        self.assertIn("custom_name_fusion", model.adapters_config.fusions)
        self.assertIn("custom_name_fusion", model.adapters_config.fusion_name_map)

        # check forward pass
        model.set_active_adapters(Fuse(name1, name2, name="custom_name_fusion"))
        fusion_custom_output = model(**input_data)
        model.set_active_adapters(Fuse(name1, name2))
        fusion_default_output = model(**input_data)
        model.set_active_adapters(None)
        base_output = model(**input_data)

        self.assertFalse(torch.equal(fusion_default_ref_output[0], base_output[0]))
        self.assertTrue(torch.equal(fusion_default_ref_output[0], fusion_default_output[0]))
        self.assertFalse(torch.equal(fusion_custom_output[0], fusion_default_output[0]))
        self.assertFalse(torch.equal(fusion_custom_output[0], base_output[0]))

        # delete only the custom fusion
        model.delete_adapter_fusion(Fuse(name1, name2, name="custom_name_fusion"))
        # model.delete_adapter_fusion("custom_name_fusion")

        self.assertIn(f"{name1},{name2}", model.adapters_config.fusions)
        self.assertNotIn("custom_name_fusion", model.adapters_config.fusions)

    def test_load_adapter_fusion_custom_name(self):
        model1 = self.get_model()
        model1.eval()

        name1 = "name1"
        name2 = "name2"
        model1.add_adapter(name1)
        model1.add_adapter(name2)

        model2 = copy.deepcopy(model1)
        model2.eval()

        model1.add_adapter_fusion([name1, name2], name="custom_name_fusion")
        model1.set_active_adapters(Fuse(name1, name2, name="custom_name_fusion"))

        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter_fusion(temp_dir, "custom_name_fusion")
            # also tests that set_active works
            model2.load_adapter_fusion(temp_dir, set_active=True)

        # check if adapter was correctly loaded
        self.assertEqual(model1.adapters_config.fusions.keys(), model2.adapters_config.fusions.keys())

        # check equal output
        in_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**in_data)
        output2 = model2(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
