from transformers import BeitConfig

from .generator import VisionAdapterTestBase, generate_method_tests, make_config


class BeitAdapterTestBase(VisionAdapterTestBase):
    config_class = BeitConfig
    config = make_config(
        BeitConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = "microsoft/beit-base-patch16-224-pt22k"


method_tests = generate_method_tests(BeitAdapterTestBase, excluded_tests=["Composition", "Embeddings"])

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
