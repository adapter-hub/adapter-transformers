from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPVisionModelWithProjection

from .generator import *


class CLIPVisionAdapterTestBase(VisionAdapterTestBase):
    model_class = CLIPVisionModel
    config_class = CLIPVisionConfig
    config = make_config(
        CLIPVisionConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = "openai/clip-vit-base-patch32"


@require_torch
class CLIPVisionAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    CLIPVisionAdapterTestBase,
    unittest.TestCase,
):
    pass


class CLIPVisionWithProjectionAdapterTestBase(VisionAdapterTestBase):
    model_class = CLIPVisionModelWithProjection
    config_class = CLIPVisionConfig
    config = make_config(
        CLIPVisionConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = "openai/clip-vit-base-patch32"


@require_torch
class CLIPVisionWithProjectionAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    CLIPVisionWithProjectionAdapterTestBase,
    unittest.TestCase,
):
    pass
