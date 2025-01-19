from tests.test_methods.generator import *
from transformers import CLIPVisionConfig, CLIPVisionModel


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


method_tests = generate_method_tests(
    model_test_base=CLIPVisionAdapterTestBase,
    not_supported=["Embeddings", "Heads", "Composition", "ClassConversion", "PromptTuning", "ConfigUnion"],
)


for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
