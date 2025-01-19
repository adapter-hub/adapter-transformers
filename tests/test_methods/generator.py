import inspect
import sys
import unittest
from math import ceil

import pytest
import torch

from tests.test_methods.base import AudioAdapterTestBase, TextAdapterTestBase, VisionAdapterTestBase
from tests.test_methods.method_test_impl.composition.test_parallel import (
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
)
from tests.test_methods.method_test_impl.core.test_adapter_backward_compability import CompabilityTestMixin
from tests.test_methods.method_test_impl.core.test_adapter_conversion import ModelClassConversionTestMixin
from tests.test_methods.method_test_impl.core.test_adapter_fusion_common import AdapterFusionModelTestMixin
from tests.test_methods.method_test_impl.embeddings.test_adapter_embeddings import EmbeddingTestMixin
from tests.test_methods.method_test_impl.heads.test_adapter_heads import PredictionHeadModelTestMixin
from tests.test_methods.method_test_impl.peft.test_adapter_common import BottleneckAdapterTestMixin
from tests.test_methods.method_test_impl.peft.test_compacter import CompacterTestMixin
from tests.test_methods.method_test_impl.peft.test_config_union import ConfigUnionAdapterTest
from tests.test_methods.method_test_impl.peft.test_ia3 import IA3TestMixin
from tests.test_methods.method_test_impl.peft.test_lora import LoRATestMixin
from tests.test_methods.method_test_impl.peft.test_prefix_tuning import PrefixTuningTestMixin
from tests.test_methods.method_test_impl.peft.test_prompt_tuning import PromptTuningTestMixin
from tests.test_methods.method_test_impl.peft.test_reft import ReftTestMixin
from tests.test_methods.method_test_impl.peft.test_unipelt import UniPELTTestMixin
from tests.test_methods.method_test_impl.utils import make_config
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.testing_utils import require_torch, torch_device


def generate_method_tests(
    model_test_base,
    redundant=[],
    not_supported=[],
) -> dict:
    """
    Generates a set of method test classes for a given model test base.

    Args:
        model_test_base (type): The base class for the model tests.
        redundant (list, optional): A list of redundant tests to exclude. Defaults to [].
        not_supported (list, optional): A list of tests that are not supported for the model. Defaults to [].

    Returns:
        dict: A dictionary mapping test class names to the generated test classes.
    """
    test_classes = {}

    if "Core" not in redundant and "Core" not in not_supported:

        @require_torch
        @pytest.mark.core
        class Core(
            model_test_base,
            CompabilityTestMixin,
            AdapterFusionModelTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["Core"] = Core

    if "Heads" not in redundant and "Heads" not in not_supported:

        @require_torch
        @pytest.mark.heads
        class Heads(
            model_test_base,
            PredictionHeadModelTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["Heads"] = Heads

    if "Embeddings" not in redundant and "Embeddings" not in not_supported:

        @require_torch
        @pytest.mark.embeddings
        class Embeddings(
            model_test_base,
            EmbeddingTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["Embeddings"] = Embeddings

    if "Composition" not in redundant and "Composition" not in not_supported:

        @require_torch
        @pytest.mark.composition
        class Composition(
            model_test_base,
            ParallelAdapterInferenceTestMixin,
            ParallelTrainingMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["Composition"] = Composition

    if "ClassConversion" not in redundant and "ClassConversion" not in not_supported:

        @require_torch
        class ClassConversion(
            ModelClassConversionTestMixin,
            model_test_base,
            unittest.TestCase,
        ):
            pass

        test_classes["ClassConversion"] = ClassConversion

    if "PrefixTuning" not in redundant and "PrefixTuning" not in not_supported:

        @require_torch
        @pytest.mark.prefix_tuning
        class PrefixTuning(
            model_test_base,
            PrefixTuningTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["PrefixTuning"] = PrefixTuning

    if "PromptTuning" not in redundant and "PromptTuning" not in not_supported:

        @require_torch
        @pytest.mark.prompt_tuning
        class PromptTuning(
            model_test_base,
            PromptTuningTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["PromptTuning"] = PromptTuning

    if "ReFT" not in redundant and "ReFT" not in not_supported:

        @require_torch
        @pytest.mark.reft
        class ReFT(
            model_test_base,
            ReftTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["ReFT"] = ReFT

    if "UniPELT" not in redundant and "UniPELT" not in not_supported:

        @require_torch
        @pytest.mark.unipelt
        class UniPELT(
            model_test_base,
            UniPELTTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["UniPELT"] = UniPELT

    if "Compacter" not in redundant and "Compacter" not in not_supported:

        @require_torch
        @pytest.mark.compacter
        class Compacter(
            model_test_base,
            CompacterTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["Compacter"] = Compacter

    if "Bottleneck" not in redundant and "Bottleneck" not in not_supported:

        @require_torch
        @pytest.mark.bottleneck
        class Bottleneck(
            model_test_base,
            BottleneckAdapterTestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["Bottleneck"] = Bottleneck

    if "IA3" not in redundant and "IA3" not in not_supported:

        @require_torch
        @pytest.mark.ia3
        class IA3(
            model_test_base,
            IA3TestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["IA3"] = IA3

    if "LoRA" not in redundant and "LoRA" not in not_supported:

        @require_torch
        @pytest.mark.lora
        class LoRA(
            model_test_base,
            LoRATestMixin,
            unittest.TestCase,
        ):
            pass

        test_classes["LoRA"] = LoRA

    if "ConfigUnion" not in redundant and "ConfigUnion" not in not_supported:

        @require_torch
        @pytest.mark.config_union
        class ConfigUnion(
            model_test_base,
            ConfigUnionAdapterTest,
            unittest.TestCase,
        ):
            pass

        test_classes["ConfigUnion"] = ConfigUnion

    return test_classes
