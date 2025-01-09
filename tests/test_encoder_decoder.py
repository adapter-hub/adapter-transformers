# flake8: noqa: F403,F405
import unittest

import adapters
from hf_transformers.tests.models.encoder_decoder.test_modeling_encoder_decoder import *  # Imported to execute model tests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertConfig
from transformers.testing_utils import require_torch, torch_device

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase
from .test_adapter_fusion_common import AdapterFusionModelTestMixin


class EncoderDecoderAdapterTestBase(AdapterTestBase):
    model_class = EncoderDecoderModel
    config_class = EncoderDecoderConfig
    config = staticmethod(
        lambda: EncoderDecoderConfig.from_encoder_decoder_configs(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            ),
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
                is_decoder=True,
                add_cross_attention=True,
            ),
        )
    )
    tokenizer_name = "bert-base-uncased"
    do_run_train_tests = False


@require_torch
class EncoderDecoderAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    EncoderDecoderAdapterTestBase,
    unittest.TestCase,
):
    def test_generation(self):
        model = AutoModelForSeq2SeqLM.from_config(self.config())
        adapters.init(model)
        model.add_adapter("test", config="pfeiffer")
        model.set_active_adapters("test")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)

        text = "This is a test sentence."
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        generated_ids = model.generate(input_ids, bos_token_id=100)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.assertNotEqual("", generated_text)

    def test_invertible_adapter_with_head(self):
        """This test class is copied and adapted from the identically-named test in test_adapter_heads.py."""
        raise self.skipTest("AutoModelForSeq2SeqLM does not support using invertible adapters.")

    def test_adapter_fusion_save_with_head(self):
        # This test is not applicable to the encoder-decoder model since it has no heads.
        self.skipTest("Not applicable to the encoder-decoder model.")

    def test_forward_with_past(self):
        # This test is not applicable to the encoder-decoder model since it has no heads.
        self.skipTest("Not applicable to the encoder-decoder model.")

    def test_output_adapter_gating_scores_unipelt(self):
        # TODO currently not supported
        self.skipTest("Not implemented.")

    def test_output_adapter_fusion_attentions(self):
        # TODO currently not supported
        self.skipTest("Not implemented.")
