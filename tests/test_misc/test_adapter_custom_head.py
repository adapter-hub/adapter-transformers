import tempfile
import unittest

import torch

from adapters import AutoAdapterModel
from adapters.heads import ClassificationHead, PredictionHead
from tests.test_methods.method_test_impl.utils import ids_tensor
from transformers import AutoConfig
from transformers.testing_utils import require_torch, torch_device


class CustomHead(PredictionHead):
    def __init__(
        self,
        model,
        head_name,
        **config,
    ):
        super().__init__(head_name)
        self.config = config
        self.build(model=model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        logits = super().forward(outputs[0])
        outputs = (logits,) + outputs[2:]
        return outputs


@require_torch
class AdapterCustomHeadTest(unittest.TestCase):
    def test_add_custom_head(self):
        model_name = "bert-base-uncased"
        model = AutoAdapterModel.from_pretrained(model_name)
        model.register_custom_head("tag", CustomHead)
        config = {"num_labels": 3, "layers": 2, "activation_function": "tanh"}
        model.add_custom_head(head_type="tag", head_name="custom_head", **config)
        model.eval()
        model.to(torch_device)
        in_data = ids_tensor((1, 128), 1000)
        output1 = model(in_data)
        model.add_tagging_head("tagging_head", num_labels=3, layers=2)
        model.to(torch_device)
        output2 = model(in_data)
        self.assertEqual(output1[0].size(), output2[0].size())

    def test_save_load_custom_head(self):
        model_name = "bert-base-uncased"
        model_config = AutoConfig.from_pretrained(model_name)
        model1 = AutoAdapterModel.from_pretrained(model_name, config=model_config)
        model2 = AutoAdapterModel.from_pretrained(model_name, config=model_config)
        model1.custom_heads = {"tag": CustomHead}
        model2.custom_heads = {"tag": CustomHead}
        config = {"num_labels": 3, "layers": 2, "activation_function": "tanh"}
        model1.add_custom_head(head_type="tag", head_name="custom_head", **config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_head(temp_dir, "custom_head")
            model2.load_head(temp_dir)

        model1.eval()
        model2.eval()

        in_data = ids_tensor((1, 128), 1000)
        model1.to(torch_device)
        model2.to(torch_device)
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(output1[0].size(), output2[0].size())
        state1 = model1.heads["custom_head"].state_dict()
        state2 = model2.heads["custom_head"].state_dict()
        for (k1, v1), (k2, v2) in zip(state1.items(), state2.items()):
            self.assertTrue(torch.equal(v1, v2))

    def test_builtin_head_as_custom(self):
        model_name = "bert-base-uncased"
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.custom_heads = {"tag": CustomHead}
        model = AutoAdapterModel.from_pretrained(model_name, config=model_config)
        in_data = ids_tensor((1, 128), 1000)

        model.register_custom_head("classification", ClassificationHead)
        model.add_custom_head(
            head_type="classification", head_name="custom_head", num_labels=3, layers=2, activation_function="tanh"
        )
        model.eval()
        model.to(torch_device)
        output = model(in_data)

        self.assertEqual((1, 3), output[0].shape)
        self.assertEqual("custom_head", model.active_head)
