import transformers
from check_repo import get_models
from transformers import ModelAdaptersMixin


MODELS_WITH_ADAPTERS = [
    "bert",
    "roberta",
    "xlm_roberta",
    "distilbert",
    "bart",
    "mbart",
    "gpt2",
    "encoder_decoder",
    "t5",
    "deberta",
    "deberta-v2",
    "transformer",
]

IGNORE_NOT_IMPLEMENTING_MIXIN = [
    "BartEncoder",
    "BartDecoder",
    "MBartEncoder",
    "MBartDecoder",
    "T5Stack",
]


def check_models_implement_mixin():
    """Checks that all model classes belonging to modules that have adapter-support implemented properly derive the adapter mixin."""
    failures = []
    for model in dir(transformers.models):
        if model in MODELS_WITH_ADAPTERS:
            model_module = getattr(transformers.models, model)
            for submodule in dir(model_module):
                if submodule.startswith("modeling"):
                    modeling_module = getattr(model_module, submodule)
                    for model_name, model_class in get_models(modeling_module):
                        if (
                            not issubclass(model_class, ModelAdaptersMixin)
                            and model_name not in IGNORE_NOT_IMPLEMENTING_MIXIN
                        ):
                            failures.append(f"{model_name} should implement ModelAdaptersMixin.")
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    print("Checking all adapter-supporting models implement mixin.")
    check_models_implement_mixin()
