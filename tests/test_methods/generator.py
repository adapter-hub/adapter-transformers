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
    excluded_tests=[],
) -> dict:
    """
    Generates a set of method test classes for a given model test base.

    Args:
        model_test_base (type): The base class for the model tests.
        excluded_mixins (list, optional): A list of mixin classes to exclude from the test classes.

    Returns:
        dict: A dictionary mapping test class names to the generated test classes.
    """
    test_classes = {}

    @require_torch
    @pytest.mark.core
    class Core(
        model_test_base,
        CompabilityTestMixin,
        AdapterFusionModelTestMixin,
        unittest.TestCase,
    ):
        pass

    if "Core" not in excluded_tests:
        test_classes["Core"] = Core

    @require_torch
    @pytest.mark.heads
    class Heads(
        model_test_base,
        PredictionHeadModelTestMixin,
        unittest.TestCase,
    ):
        pass

    if "Heads" not in excluded_tests:
        test_classes["Heads"] = Heads

    @require_torch
    @pytest.mark.embeddings
    class Embeddings(
        model_test_base,
        EmbeddingTestMixin,
        unittest.TestCase,
    ):
        pass

    if "Embeddings" not in excluded_tests:
        test_classes["Embeddings"] = Embeddings

    @require_torch
    @pytest.mark.composition
    class Composition(
        model_test_base,
        ParallelAdapterInferenceTestMixin,
        ParallelTrainingMixin,
        unittest.TestCase,
    ):
        pass

    if "Composition" not in excluded_tests:
        test_classes["Composition"] = Composition

    @require_torch
    class ClassConversion(
        ModelClassConversionTestMixin,
        model_test_base,
        unittest.TestCase,
    ):
        pass

    if "ClassConversion" not in excluded_tests:
        test_classes["ClassConversion"] = ClassConversion

    @require_torch
    @pytest.mark.prefix_tuning
    class PrefixTuning(
        model_test_base,
        PrefixTuningTestMixin,
        unittest.TestCase,
    ):
        pass

    if "PrefixTuning" not in excluded_tests:
        test_classes["PrefixTuning"] = PrefixTuning

    @require_torch
    @pytest.mark.prompt_tuning
    class PromptTuning(
        model_test_base,
        PromptTuningTestMixin,
        unittest.TestCase,
    ):
        pass

    if "PromptTuning" not in excluded_tests:
        test_classes["PromptTuning"] = PromptTuning

    @require_torch
    @pytest.mark.reft
    class ReFT(
        model_test_base,
        ReftTestMixin,
        unittest.TestCase,
    ):
        pass

    if "ReFT" not in excluded_tests:
        test_classes["ReFT"] = ReFT

    @require_torch
    @pytest.mark.unipelt
    class UniPELT(
        model_test_base,
        UniPELTTestMixin,
        unittest.TestCase,
    ):
        pass

    if "UniPELT" not in excluded_tests:
        test_classes["UniPELT"] = UniPELT

    @require_torch
    @pytest.mark.compacter
    class Compacter(
        model_test_base,
        CompacterTestMixin,
        unittest.TestCase,
    ):
        pass

    if "Compacter" not in excluded_tests:
        test_classes["Compacter"] = Compacter

    @require_torch
    @pytest.mark.bottleneck
    class Bottleneck(
        model_test_base,
        BottleneckAdapterTestMixin,
        unittest.TestCase,
    ):
        pass

    if "Bottleneck" not in excluded_tests:
        test_classes["Bottleneck"] = Bottleneck

    @require_torch
    @pytest.mark.ia3
    class IA3(
        model_test_base,
        IA3TestMixin,
        unittest.TestCase,
    ):
        pass

    if "IA3" not in excluded_tests:
        test_classes["IA3"] = IA3

    @require_torch
    @pytest.mark.lora
    class LoRA(
        model_test_base,
        LoRATestMixin,
        unittest.TestCase,
    ):
        pass

    if "LoRA" not in excluded_tests:
        test_classes["LoRA"] = LoRA

    @require_torch
    @pytest.mark.config_union
    class ConfigUnion(
        model_test_base,
        ConfigUnionAdapterTest,
        unittest.TestCase,
    ):
        pass

    if "ConfigUnion" not in excluded_tests:
        test_classes["ConfigUnion"] = ConfigUnion

    return test_classes
