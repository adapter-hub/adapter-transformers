from transformers import ViTConfig

from .generator import *


class ViTAdapterTestBase(VisionAdapterTestBase):
    config_class = ViTConfig
    config = make_config(
        ViTConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = "google/vit-base-patch16-224-in21k"


method_tests = generate_method_tests(ViTAdapterTestBase, not_supported=["ConfigUnion", "Embeddings", "Composition"])
for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
