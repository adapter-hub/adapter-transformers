import json
import os
import tempfile

from adapters import SeqBnConfig, __version__
from tests.test_methods.method_test_impl.utils import create_twin_models
from transformers.testing_utils import require_torch


@require_torch
class CompabilityTestMixin:
    def test_load_old_non_linearity(self):
        model1, model2 = create_twin_models(self.model_class, self.config)
        config = SeqBnConfig(non_linearity="gelu")
        name = "dummy"
        model1.add_adapter(name, config=config)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            with open(os.path.join(temp_dir, "adapter_config.json"), "r") as file:
                data = json.load(file)
                data["config"]["non_linearity"] = "gelu_orig"
                del data["version"]
            with open(os.path.join(temp_dir, "adapter_config.json"), "w") as file:
                json.dump(data, file)

            # also tests that set_active works
            model2.load_adapter(temp_dir, set_active=True)

        # check if adapter was correctly loaded
        self.assertTrue(name in model2.adapters_config)
        self.assertEqual(
            "gelu", model2.adapters_config.config_map[model2.adapters_config.adapters[name]]["non_linearity"]
        )

    def test_save_version_with_adapter(self):
        model = self.get_model()
        config = SeqBnConfig(non_linearity="gelu")
        name = "dummy"
        model.add_adapter(name, config=config)
        model.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_adapter(temp_dir, name)

            with open(os.path.join(temp_dir, "adapter_config.json"), "r") as file:
                data = json.load(file)
                self.assertEqual(__version__, data["version"].replace("adapters.", ""))
