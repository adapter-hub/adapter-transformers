# flake8: noqa: F403,F405
import numpy as np

from adapters import CLIPAdapterModel
from hf_transformers.tests.models.clip.test_modeling_clip import *  # Imported to execute model tests
from hf_transformers.tests.test_modeling_common import _config_zero_init
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class CLIPAdapterModelTest(AdapterModelTesterMixin, CLIPModelTest):
    all_model_classes = (CLIPAdapterModel,)
    fx_compatible = False

    # override as the `logit_scale` parameter has a different name in the adapter model
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "clip.logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
