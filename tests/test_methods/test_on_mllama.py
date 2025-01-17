import os
from pathlib import Path

from PIL import Image

from transformers import MllamaImageProcessor
from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaTextConfig, MllamaVisionConfig

from .generator import *


def from_text_vision_configs(config_class, text_config: MllamaTextConfig, vision_config: MllamaVisionConfig, **kwargs):
    """
    Create a MllamaConfig instance from separate text and vision configs.

    This standalone function mimics the behavior of class methods like CLIPConfig.from_text_vision_configs,
    but works without modifying the MllamaConfig class.

    Args:
        config_class: The configuration class to instantiate (MllamaConfig)
        text_config: The configuration for the text model
        vision_config: The configuration for the vision model
        **kwargs: Additional arguments to pass to the config constructor

    Returns:
        An instance of config_class initialized with the text and vision configs
    """
    return config_class(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


class MllamaAdapterTestBase(TextAdapterTestBase):

    config = staticmethod(
        lambda: from_text_vision_configs(
            MllamaConfig,
            MllamaTextConfig(
                vocab_size=1000,  # Minimal vocab size
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                intermediate_size=256,
                cross_attention_layers=[0],
                bos_token_id=990,
                eos_token_id=991,
                pad_token_id=992,
                max_position_embeddings=512,
                rope_scaling={
                    "rope_type": "default",
                },
            ),
            MllamaVisionConfig(
                hidden_size=128,
                num_hidden_layers=4,
                num_global_layers=4,
                num_attention_heads=1,
                intermediate_size=256,
                vision_output_dim=128,
                image_size=224,
            ),
        )
    )
    tokenizer_name = "arnavgrg/mllama-11b-vision-lora"
    shape = (1, 128)

    # Save runtime by computing the processed image once and reusing it for all tests
    FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

    img_processor = MllamaImageProcessor()
    img = Image.open(os.path.join(FIXTURES_DIR, "tests_samples", "COCO", "000000039769.png"))
    processed_img = img_processor(img, return_tensors="pt")

    def get_input_samples(self, vocab_size=1000, shape=None, config=None, dtype=torch.float, **kwargs):
        shape = shape or self.input_shape

        # Text inputs
        input_ids = self.build_rand_ids_tensor(shape, vocab_size)

        in_data = {
            "input_ids": input_ids,
            "pixel_values": self.processed_img["pixel_values"],
            "aspect_ratio_ids": self.processed_img["aspect_ratio_ids"],
            "aspect_ratio_mask": self.processed_img["aspect_ratio_mask"],
        }

        if "num_labels" in kwargs:
            in_data["labels"] = self.build_rand_ids_tensor(shape[:-1], vocab_size=kwargs["num_labels"])

        return in_data


test_methods = generate_method_tests(MllamaAdapterTestBase, excluded_tests=[])

for test_class_name, test_class in test_methods.items():
    globals()[test_class_name] = test_class


""" resources:
https://github.com/AdrianBZG/llama-multimodal-vqa
https://huggingface.co/blog/llama32
https://github.com/huggingface/huggingface-llama-recipes/blob/main/fine_tune/Llama-Vision%20FT.ipynb
"""
