import tempfile
import unittest

import torch

import adapters
from adapters import AdapterModelInterface, AdapterSetup, DoubleSeqBnConfig, LoRAConfig, ParBnConfig, load_model
from transformers import Gemma2ForCausalLM, Gemma2ForSequenceClassification
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.testing_utils import require_torch, torch_device

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    ReftTestMixin,
    create_twin_models,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin


class CustomInterfaceModelTestBase(AdapterTestBase):
    model_class = Gemma2ForCausalLM
    config_class = Gemma2Config
    config = make_config(
        Gemma2Config,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=16,
        pad_token_id=0,
    )
    tokenizer_name = "yujiepan/gemma-2-tiny-random"
    adapter_interface = AdapterModelInterface(
        adapter_types=["bottleneck", "lora", "reft", "invertible"],
        model_embeddings="embed_tokens",
        model_layers="layers",
        layer_self_attn="self_attn",
        layer_cross_attn=None,
        attn_k_proj="k_proj",
        attn_q_proj="q_proj",
        attn_v_proj="v_proj",
        attn_o_proj="o_proj",
        layer_intermediate_proj="mlp.up_proj",
        layer_output_proj="mlp.down_proj",
        layer_pre_self_attn="input_layernorm",
        layer_pre_cross_attn=None,
        layer_pre_ffn="pre_feedforward_layernorm",
        layer_ln_1="post_attention_layernorm",
        layer_ln_2="post_feedforward_layernorm",
    )

    def get_model(self):
        model = Gemma2ForCausalLM(self.config())
        adapters.init(model, interface=self.adapter_interface)
        model.to(torch_device)
        return model


@require_torch
class CustomInterfaceModelTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    # PrefixTuningTestMixin,
    # PromptTuningTestMixin,
    ReftTestMixin,
    # UniPELTTestMixin,
    EmbeddingTestMixin,
    AdapterFusionModelTestMixin,
    # ParallelAdapterInferenceTestMixin,
    # ParallelTrainingMixin,
    CustomInterfaceModelTestBase,
    unittest.TestCase,
):
    # Modify the list here since we don't support MAMConfig (due to prefix tuning)
    adapter_configs_to_test = [
        (DoubleSeqBnConfig(), ["adapters.{name}."]),
        (ParBnConfig(init_weights="bert"), ["adapters.{name}."]),
    ]

    def create_twin_models(self):
        return create_twin_models(self.model_class, self.config, self.adapter_interface)

    # Copied from base.py to pass custom interface to load_model
    def run_full_model_load_test(self, adapter_config):
        model1 = self.get_model()
        model1.eval()

        name = "dummy"
        model1.add_adapter(name, config=adapter_config)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_pretrained(temp_dir)

            model2, loading_info = load_model(
                temp_dir, self.model_class, output_loading_info=True, interface=self.adapter_interface
            )

        # check if all weights were loaded
        self.assertEqual(0, len(loading_info["missing_keys"]), loading_info["missing_keys"])
        self.assertEqual(0, len(loading_info["unexpected_keys"]), loading_info["unexpected_keys"])

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.adapters_config)

        # check equal output
        input_data = self.get_input_samples(config=model1.config)
        model1.to(torch_device)
        model2.to(torch_device)
        with AdapterSetup(name):
            output1 = model1(**input_data)
            output2 = model2(**input_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.allclose(output1[0], output2[0], atol=1e-4))

    def _init_model_for_train_run(self, trained_adapter_name, frozen_adapter_name, adapter_config=None):
        model = Gemma2ForSequenceClassification(self.config())
        adapters.init(model, interface=self.adapter_interface)

        model.add_adapter(trained_adapter_name, config=adapter_config or LoRAConfig(init_weights="bert"))
        model.add_adapter(frozen_adapter_name, config=adapter_config or LoRAConfig(init_weights="bert"))

        return model

    def test_load_mam_adapter(self):
        self.skipTest("Does not support prefix tuning.")

    def test_train_mam_adapter(self):
        self.skipTest("Does not support prefix tuning.")

    def test_merging_with_other_adapters(self):
        self.skipTest("Does not support all required methods yet.")

    def test_supports_adapter(self):
        model = self.get_model()
        model.eval()

        config = "unipelt"
        with self.assertRaises(ValueError):
            model.add_adapter("my_adapter", config=config)
