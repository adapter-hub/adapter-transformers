import warnings
from collections import OrderedDict

from ....models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from ....models.auto.configuration_auto import CONFIG_MAPPING_NAMES


# Make sure that children are placed before parents!
ADAPTER_MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("xlm-roberta", "XLMRobertaAdapterModel"),
        ("roberta", "RobertaAdapterModel"),
        ("bert", "BertAdapterModel"),
        ("distilbert", "DistilBertAdapterModel"),
        ("deberta-v2", "DebertaV2AdapterModel"),
        ("deberta", "DebertaAdapterModel"),
        ("bart", "BartAdapterModel"),
        ("mbart", "MBartAdapterModel"),
        ("gpt2", "GPT2AdapterModel"),
        ("t5", "T5AdapterModel"),
        ("vit", "ViTAdapterModel"),
    ]
)
MODEL_WITH_HEADS_MAPPING_NAMES = OrderedDict(
    [
        ("xlm-roberta", "XLMRobertaModelWithHeads"),
        ("roberta", "RobertaModelWithHeads"),
        ("bert", "BertModelWithHeads"),
        ("distilbert", "DistilBertModelWithHeads"),
        ("bart", "BartModelWithHeads"),
        ("mbart", "MBartModelWithHeads"),
        ("gpt2", "GPT2ModelWithHeads"),
        ("t5", "T5ModelWithHeads"),
    ]
)

ADAPTER_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, ADAPTER_MODEL_MAPPING_NAMES)
MODEL_WITH_HEADS_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_HEADS_MAPPING_NAMES)


class AutoAdapterModel(_BaseAutoModelClass):
    _model_mapping = ADAPTER_MODEL_MAPPING


AutoAdapterModel = auto_class_update(AutoAdapterModel, head_doc="adapters and flexible heads")


class AutoModelWithHeads(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_HEADS_MAPPING

    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "This class has been renamed to `{}` in v3. "
            "Please use the new class instead as this class might be removed in a future version.".format(
                cls.__bases__[0].__name__
            ),
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


AutoModelWithHeads = auto_class_update(AutoModelWithHeads, head_doc="flexible heads")
