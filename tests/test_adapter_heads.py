import tempfile

import torch

from transformers import MODEL_WITH_HEADS_MAPPING, AutoModelWithHeads
from transformers.testing_utils import require_torch, torch_device

from .test_adapter_common import create_twin_models


@require_torch
class PredictionHeadModelTestMixin:

    batch_size = 1
    seq_length = 128

    def run_prediction_head_test(
        self, model, compare_model, head_name, input_shape=None, output_shape=(1, 2), label_dict=None
    ):
        # first, check if the head is actually correctly registered as part of the pt module
        self.assertTrue(f"heads.{head_name}" in dict(model.named_modules()))

        # save & reload
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir, head_name)

            compare_model.load_head(temp_dir)

        # check if adapter was correctly loaded
        self.assertTrue(head_name in compare_model.heads)

        # make a forward pass
        model.active_head = head_name
        input_shape = input_shape or (self.batch_size, self.seq_length)
        in_data = {"input_ids": self.get_input_samples(input_shape, config=model.config)}
        if label_dict:
            for k, v in label_dict.items():
                in_data[k] = v
        output1 = model(**in_data)
        self.assertEqual(output_shape, tuple(output1[1].size()))
        # check equal output
        compare_model.active_head = head_name
        output2 = compare_model(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[1], output2[1]))

    def test_classification_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_classification_head"):
            return

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_classification_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(model1, model2, "dummy", label_dict=label_dict)

    def test_multiple_choice_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_multiple_choice_head"):
            return

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_multiple_choice_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.ones(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", input_shape=(self.batch_size, 2, self.seq_length), label_dict=label_dict
        )

    def test_tagging_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_tagging_head"):
            return

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_tagging_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(model1, model2, "dummy", output_shape=(1, 128, 2), label_dict=label_dict)

    def test_qa_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_qa_head"):
            return

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_qa_head("dummy")
        label_dict = {}
        label_dict["start_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        label_dict["end_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(model1, model2, "dummy", output_shape=(1, 128), label_dict=label_dict)

    def test_adapter_with_head(self):
        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        name = "dummy"
        model1.add_adapter(name)
        model1.add_classification_head(name, num_labels=3)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            model2.load_adapter(temp_dir)
            model2.set_active_adapters(name)
        # check equal output
        in_data = self.get_input_samples((1, 128), config=model1.config)
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(3, output1[0].size()[1])

    def test_adapter_with_head_load_as(self):
        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        name = "dummy"
        model1.add_adapter(name)
        model1.add_classification_head(name, num_labels=3)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            # reload using a different name
            model2.load_adapter(temp_dir, load_as="new_name")
            model2.set_active_adapters("new_name")

        # check equal output
        in_data = self.get_input_samples((1, 128), config=model1.config)
        output1 = model1(in_data)
        output2 = model2(in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(3, output1[0].size()[1])

    def test_load_full_model(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.add_classification_head("dummy", layers=1)

        true_config = model.get_prediction_heads_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            # save
            model.save_pretrained(temp_dir)
            # reload
            model = AutoModelWithHeads.from_pretrained(temp_dir)
        self.assertIn("dummy", model.heads)
        self.assertDictEqual(true_config, model.get_prediction_heads_config())
