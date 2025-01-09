import copy
import os
import tempfile

import torch

import adapters
from adapters import (
    ADAPTER_CONFIG_MAP,
    ADAPTER_MODEL_MAPPING,
    AutoAdapterModel,
    BatchSplit,
    DoubleSeqBnConfig,
    DoubleSeqBnInvConfig,
    Fuse,
    MAMConfig,
    SeqBnConfig,
    SeqBnInvConfig,
    Stack,
)
from adapters.heads.language_modeling import CausalLMHead
from adapters.utils import SETUP_CONFIG_NAME
from transformers import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, CLIPConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin, create_twin_models


@require_torch
class BottleneckAdapterTestMixin(AdapterMethodBaseTestMixin):
    adapter_configs_to_test = [
        (SeqBnConfig(), ["adapters.{name}."]),
        (MAMConfig(), ["adapters.{name}.", "prefix_tunings.{name}."]),
    ]

    inv_adapter_configs_to_test = [
        (SeqBnInvConfig(), ["invertible_adapters.{name}"]),
        (DoubleSeqBnInvConfig(), ["invertible_adapters.{name}"]),
    ]

    def test_add_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_leave_out_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_leave_out_test(model, adapter_config, self.leave_out_layers)

    def test_linear_average_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_linear_average_test(model, adapter_config, filter_keys)

    def test_delete_adapter(self):
        model = self.get_model()
        model.eval()

        for adapter_config, filter_keys in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_delete_test(model, adapter_config, filter_keys)

    def test_add_adapter_with_invertible(self):
        model = self.get_model().base_model
        model.eval()
        if not model.supports_adapter("invertible"):
            self.skipTest("Model does not support invertible adapters.")

        for adapter_config in [SeqBnInvConfig(), DoubleSeqBnInvConfig()]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters(name)

                # adapter is correctly added to config
                self.assertTrue(name in model.adapters_config)
                self.assertEqual(adapter_config, model.adapters_config.get(name))

                # invertible adapter is correctly added and returned
                self.assertTrue(name in model.invertible_adapters)
                self.assertEqual(model.invertible_adapters[name], model.get_invertible_adapter())

                # all invertible adapter weights should be activated for training
                for param in model.invertible_adapters[name].parameters():
                    self.assertTrue(param.requires_grad)

                # Set a hook before the invertible adapter to make sure it's actually called twice:
                # Once after the embedding layer and once in the prediction head.
                calls = 0

                def forward_pre_hook(module, input):
                    nonlocal calls
                    calls += 1

                model.get_invertible_adapter().register_forward_pre_hook(forward_pre_hook)

                # check forward pass
                input_data = self.get_input_samples(config=model.config)
                model.to(torch_device)
                adapter_output = model(**input_data)
                # make sure the output is different without invertible adapter
                del model.invertible_adapters[name]
                self.assertFalse(name in model.invertible_adapters)
                adapter_output_no_inv = model(**input_data)
                self.assertEqual(len(adapter_output), len(adapter_output_no_inv))
                self.assertFalse(torch.equal(adapter_output[0], adapter_output_no_inv[0]))
                # We expect one call to invertible adapter
                self.assertEqual(1, calls)

    def test_delete_adapter_with_invertible(self):
        """Tests if the invertible adapters are deleted correctly."""
        model = self.get_model().base_model
        model.eval()
        if not model.supports_adapter("invertible"):
            self.skipTest("Model does not support invertible adapters.")

        # iterate through all adapter invertible adapter configs
        for adapter_config, filter_keys in self.inv_adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters(name)

                # check if adapter is correctly added to config
                self.assert_adapter_available(model, name)
                # remove the adapter again
                model.delete_adapter(name)

                # check if adapter is correctly removed from the model
                self.assert_adapter_unavailable(model, name)

                # check additionally if invertible adapter is removed correctly from the model
                self.assertFalse(name in model.invertible_adapters)
                self.assertEqual(None, model.get_invertible_adapter())

                # check that weights are available and active
                has_weights = False
                filter_keys = [k.format(name=name) for k in filter_keys]
                for k, v in self.filter_parameters(model, filter_keys).items():
                    has_weights = True
                self.assertFalse(has_weights)

    def test_get_adapter(self):
        model = self.get_model()
        model.eval()
        n_layers = len(list(model.iter_layers()))
        if model.config.is_encoder_decoder:
            n_prefix_layers = 3
        elif model.config.is_composition or isinstance(model.config, CLIPConfig):
            n_prefix_layers = 2
        else:
            n_prefix_layers = 1

        for adapter_config, n_expected in [
            (DoubleSeqBnConfig(), n_layers * 2),
            (MAMConfig(), n_layers + n_prefix_layers),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_get_test(model, adapter_config, n_expected)

    def test_add_adapter_multiple_reduction_factors(self):
        model = self.get_model()
        model.eval()
        reduction_factor = {"1": 1, "default": 2}
        for adapter_config in [
            SeqBnConfig(reduction_factor=reduction_factor),
            DoubleSeqBnConfig(reduction_factor=reduction_factor),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters(name)

                # adapter is correctly added to config
                self.assertTrue(name in model.adapters_config)
                self.assertEqual(adapter_config, model.adapters_config.get(name))

                adapter = model.get_adapter(name)

                self.assertEqual(
                    adapter[0]["output_adapter"].adapter_down[0].in_features
                    / adapter[0]["output_adapter"].adapter_down[0].out_features,
                    reduction_factor["default"],
                )
                self.assertEqual(
                    adapter[1]["output_adapter"].adapter_down[0].in_features
                    / adapter[1]["output_adapter"].adapter_down[0].out_features,
                    reduction_factor["1"],
                )

    def test_reduction_factor_no_default(self):
        model = self.get_model()
        model.eval()
        reduction_factor = {"2": 8, "4": 32}
        for adapter_config in [
            SeqBnConfig(reduction_factor=reduction_factor),
            DoubleSeqBnConfig(reduction_factor=reduction_factor),
        ]:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                name = adapter_config.__class__.__name__
                with self.assertRaises(KeyError):
                    model.add_adapter(name, config=adapter_config)

    def test_forward_bottleneck(self):
        model = self.get_model()
        model.eval()

        for adapter_config, _ in self.adapter_configs_to_test:
            for dtype in self.dtypes_to_test:
                with self.subTest(
                    model_class=model.__class__.__name__, config=adapter_config.__class__.__name__, dtype=dtype
                ):
                    self.run_forward_test(model, adapter_config, dtype=dtype)

    def test_invertible_adapter_forward(self):
        model = self.get_model()
        model.eval()
        if not model.supports_adapter("invertible"):
            self.skipTest("Model does not support invertible adapters.")

        for adapter_config, _ in self.inv_adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_forward_test(model, adapter_config)

    def test_load_adapter(self):
        self.run_load_test(SeqBnConfig())

    def test_load_mam_adapter(self):
        self.run_load_test(MAMConfig())

    def test_load_full_model_adapter(self):
        self.run_full_model_load_test(SeqBnConfig())

    def test_model_config_serialization(self):
        """PretrainedConfigurations should not raise an Exception when serializing the config dict

        See, e.g., PretrainedConfig.to_json_string()
        """
        for k, v in ADAPTER_CONFIG_MAP.items():
            model = self.get_model()
            if not model.supports_adapter(v):
                continue
            # HACK: reduce the reduction factor such that
            # the small test model can have a phm_dim of 4
            if hasattr(v, "phm_layer") and v.phm_layer:
                v = v.__class__(reduction_factor=4)
            model.add_adapter("test", config=v)
            # should not raise an exception
            model.config.to_json_string()

    def test_model_adapter_summary(self):
        # count model parameters before
        model = self.get_model()
        model_no_params = sum(p.numel() for p in model.parameters())
        added = []
        for k, v in ADAPTER_CONFIG_MAP.items():
            if not model.supports_adapter(v):
                continue
            # HACK: reduce the reduction factor such that
            # the small test model can have a phm_dim of 4
            if hasattr(v, "phm_layer") and v.phm_layer:
                v = v.__class__(reduction_factor=4)
            model.add_adapter(k, config=v)
            added.append(k)
        summary = model.adapter_summary(as_dict=True)
        self.assertEqual(len(added) + 1, len(summary))
        for name in added:
            self.assertTrue(any([row["name"] == name for row in summary]))
        self.assertEqual(model_no_params, summary[-1]["#param"])

    def test_loading_adapter_weights_with_prefix(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        model_base, model_with_head = create_twin_models(self.model_class, self.config)
        model_base = model_base.base_model  # use base model w/o prefix

        model_with_head.add_adapter("dummy")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_head.save_adapter(temp_dir, "dummy")

            loading_info = {}
            model_base.load_adapter(temp_dir, loading_info=loading_info)

        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check equal output
        input_data = self.get_input_samples(config=model_with_head.config)
        model_with_head.to(torch_device)
        model_base.to(torch_device)
        output1 = model_with_head(**input_data)
        output2 = model_base(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_loading_adapter_weights_without_prefix(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")

        model_base, model_with_head = create_twin_models(self.model_class, self.config)
        model_base = model_base.base_model  # use base model w/o prefix

        model_base.add_adapter("dummy")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_base.save_adapter(temp_dir, "dummy")

            loading_info = {}
            model_with_head.load_adapter(temp_dir, loading_info=loading_info)

        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # check equal output
        input_data = self.get_input_samples(config=model_with_head.config)
        model_with_head.to(torch_device)
        model_base.to(torch_device)
        output1 = model_with_head(**input_data)
        output2 = model_base(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))

    def test_forward_with_past(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        if self.config_class not in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
            self.skipTest("No causal lm class.")

        static_model = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[self.config_class](self.config())
        adapters.init(static_model)
        flex_model = AutoAdapterModel.from_pretrained(None, config=self.config(), state_dict=static_model.state_dict())

        static_model.add_adapter("dummy")
        static_model.set_active_adapters("dummy")
        static_model.eval()
        flex_model.eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            static_model.save_adapter(temp_dir, "dummy")

            loading_info = {}
            flex_model.load_adapter(temp_dir, loading_info=loading_info)
            flex_model.set_active_adapters("dummy")

        input_data = self.get_input_samples(config=static_model.config)
        static_model.eval()
        flex_model.eval()
        static_model.to(torch_device)
        flex_model.to(torch_device)
        output = static_model(**input_data)

        input_data["past_key_values"] = output["past_key_values"]
        output_base = static_model(**input_data)
        output_with_head = flex_model(**input_data)
        self.assertTrue(torch.allclose(output_base["logits"], output_with_head["logits"]))

    def test_train_single_adapter(self):
        self.run_train_test(SeqBnConfig(), ["adapters.{name}."])

    def test_train_mam_adapter(self):
        self.run_train_test(MAMConfig(), ["adapters.{name}."])

    def test_train_adapter_fusion(self):
        if not self.do_run_train_tests:
            self.skipTest("Skipping training tests. Set `do_run_train_tests=True` to run them.")
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model = AutoAdapterModel.from_config(self.config())
        self.add_head(model, "head")

        # add the adapters to be fused
        model.add_adapter("a")
        model.add_adapter("b")
        model.add_adapter("c")

        self.assertIn("a", model.adapters_config.adapters)
        self.assertIn("b", model.adapters_config.adapters)
        self.assertIn("c", model.adapters_config.adapters)

        # setup fusion
        adapter_setup = Fuse("a", "b", "c")
        model.add_adapter_fusion(adapter_setup)
        model.train_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        self.assertEqual(adapter_setup, model.active_adapters)

        # all weights of the adapters should be frozen (test for one)
        for k, v in self.filter_parameters(model, ["adapters.a."]).items():
            self.assertFalse(v.requires_grad, k)
        # all weights of the fusion layer should be activated
        for k, v in self.filter_parameters(model, ["adapter_fusion_layer"]).items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be frozen (check on some examples)
        for k, v in self.filter_parameters(model, ["encoder.layer.0.attention"]).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        # Since our config has a value matrix, make sure it is regularized.
        # We do this by patching the fusion regularization function.
        regularization_called = False
        orig_fusion_regularization_loss = model.base_model.get_fusion_regularization_loss

        def patched_fusion_reg_loss():
            nonlocal regularization_called
            regularization_called = True
            return orig_fusion_regularization_loss()

        model.base_model.get_fusion_regularization_loss = patched_fusion_reg_loss

        self.trainings_run(model)
        self.assertTrue(regularization_called)

        def has_tied_embeddings(k):
            tied_embeddings = hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings
            is_tied_layer = (
                isinstance(model.heads["head"], CausalLMHead)
                and "heads.{}.{}.weight".format("head", len(model.heads["head"]._modules) - 1) in k
            )
            return tied_embeddings and is_tied_layer

        # check that the adapters have changed, but the base model has not
        adapters_with_change, base_with_change = False, False
        for (k1, v1), (k2, v2) in zip(state_dict_pre.items(), model.state_dict().items()):
            if (
                "adapter_fusion_layer" in k1
                or "classifier" in k1
                or "classification_head" in k1
                or "score" in k1
                or "head" in k1
            ) and not has_tied_embeddings(k1):
                adapters_with_change |= not torch.equal(v1, v2)
            else:
                base_with_change |= not torch.equal(v1, v2)
        self.assertTrue(adapters_with_change)
        self.assertFalse(base_with_change)

    def test_batch_split_training(self):
        if not self.do_run_train_tests:
            self.skipTest("Skipping training tests. Set `do_run_train_tests=True` to run them.")
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model = AutoAdapterModel.from_config(self.config())

        model.add_adapter("mrpc1")
        model.add_adapter("mrpc2")
        self.add_head(model, "mrpc1")
        self.add_head(model, "mrpc2")
        adapter_setup = BatchSplit("mrpc1", "mrpc2", batch_sizes=[1, 1])
        model.active_adapters = adapter_setup
        model.train_adapter(adapter_setup)

        # all weights of the adapter should be activated
        for k, v in self.filter_parameters(model, ["adapters.mrpc1."]).items():
            self.assertTrue(v.requires_grad, k)
        # all weights of the adapter not used for training should be frozen
        for k, v in self.filter_parameters(model, ["adapters.mrpc2."]).items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be frozen (check on some examples)
        for k, v in self.filter_parameters(model, ["encoder.layer.0.attention"]).items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        self.trainings_run(model)

        # check that the adapters have changed, but the base model has not
        adapters_with_change, base_with_change = False, False
        for (k1, v1), (k2, v2) in zip(state_dict_pre.items(), model.state_dict().items()):
            if "mrpc" in k1:
                adapters_with_change |= not torch.equal(v1, v2)
            else:
                base_with_change |= not torch.equal(v1, v2)
        self.assertTrue(adapters_with_change)
        self.assertFalse(base_with_change)

    def test_load_adapter_setup(self):
        if self.config_class not in ADAPTER_MODEL_MAPPING:
            self.skipTest("Does not support flex heads.")
        model1, model2 = create_twin_models(self.model_class, self.config)

        # Create a complex setup
        model1.add_adapter("a", config=SeqBnConfig())
        model1.add_adapter("b", config=SeqBnConfig())
        model1.add_adapter("c", config=SeqBnConfig())
        model1.add_adapter_fusion(["a", "b"])
        self.add_head(model1, "head_a")
        self.add_head(model1, "head_b")
        adapter_setup = Stack(Fuse("a", "b"), "c")
        head_setup = BatchSplit("head_a", "head_b", batch_sizes=[2, 1])
        model1.set_active_adapters(adapter_setup)
        model1.active_head = head_setup

        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter_setup(temp_dir, adapter_setup, head_setup=head_setup)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, SETUP_CONFIG_NAME)))

            # also tests that set_active works
            model2.load_adapter_setup(temp_dir, set_active=True)

        # check if adapter was correctly loaded
        for name in ["a", "b", "c"]:
            self.assertTrue(name in model2.adapters_config)
        self.assertEqual(adapter_setup, model2.active_adapters)

        # check equal output
        input_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(**input_data)
        output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0][0], output2[0][0], atol=1e-4))
        self.assertTrue(torch.allclose(output1[1][0], output2[1][0], atol=1e-4))
