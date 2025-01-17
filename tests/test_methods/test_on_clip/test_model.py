import random

from tests.test_methods.generator import *
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


class CLIPAdapterTestBase(TextAdapterTestBase):
    config_class = CLIPConfig
    config = staticmethod(
        lambda: CLIPConfig.from_text_vision_configs(
            CLIPTextConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            ),
            CLIPVisionConfig(
                image_size=224,
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            ),
        )
    )
    tokenizer_name = "openai/clip-vit-base-patch32"
    # Default shape of inputs to use
    default_text_input_samples_shape = (3, 64)
    default_vision_input_samples_shape = (3, 3, 224, 224)
    do_run_train_tests = False

    def get_input_samples(self, vocab_size=5000, config=None, dtype=torch.float, **kwargs):
        # text inputs
        shape = self.default_text_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()
        # this is needed e.g. for BART
        if config and config.eos_token_id is not None and config.eos_token_id < vocab_size:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id
        in_data = {"input_ids": input_ids}

        # vision inputs
        shape = self.default_vision_input_samples_shape
        total_dims = 1
        for dim in shape:
            total_dims *= dim
        values = []
        for _ in range(total_dims):
            values.append(random.random())
        pixel_values = torch.tensor(data=values, dtype=dtype, device=torch_device).view(shape).contiguous()
        in_data["pixel_values"] = pixel_values

        return in_data

    def add_head(self, *args, **kwargs):
        pass

    def test_adapter_fusion_save_with_head(self):
        # This test is not applicable to CLIP
        self.skipTest("Not applicable to CLIP.")

    def test_load_adapter_setup(self):
        self.skipTest("Not applicable to CLIP.")


method_tests = generate_method_tests(
    model_test_base=CLIPAdapterTestBase,
    excluded_tests=["Embeddings", "Heads", "Composition", "ClassConversion", "PromptTuning", "ConfigUnion"],
)


for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
