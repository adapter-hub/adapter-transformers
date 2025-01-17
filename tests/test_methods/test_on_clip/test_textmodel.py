from tests.test_methods.generator import *
from transformers import CLIPTextConfig, CLIPTextModel


class CLIPTextAdapterTestBase(TextAdapterTestBase):
    model_class = CLIPTextModel
    config_class = CLIPTextConfig
    config = make_config(
        CLIPTextConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "openai/clip-vit-base-patch32"


method_tests = generate_method_tests(
    model_test_base=CLIPTextAdapterTestBase,
    excluded_tests=["Embeddings", "Heads", "Composition", "ClassConversion", "PromptTuning", "ConfigUnion"],
)


for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
