# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The Adapter-Hub Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "1.1.0.dev0"

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "composition": [
        "AdapterCompositionBlock",
        "BatchSplit",
        "Fuse",
        "Parallel",
        "Split",
        "Stack",
        "parse_composition",
        "validate_composition",
    ],
    "configuration": [
        "ADAPTER_CONFIG_MAP",
        "ADAPTERFUSION_CONFIG_MAP",
        "DEFAULT_ADAPTER_CONFIG",
        "DEFAULT_ADAPTERFUSION_CONFIG",
        "AdapterConfig",
        "AdapterFusionConfig",
        "AdapterPlusConfig",
        "BnConfig",
        "CompacterConfig",
        "CompacterPlusPlusConfig",
        "ConfigUnion",
        "DiReftConfig",
        "DoubleSeqBnConfig",
        "DoubleSeqBnInvConfig",
        "DynamicAdapterFusionConfig",
        "IA3Config",
        "LoRAConfig",
        "LoReftConfig",
        "MAMConfig",
        "ModelAdaptersConfig",
        "NoReftConfig",
        "ParBnConfig",
        "PrefixTuningConfig",
        "PromptTuningConfig",
        "ReftConfig",
        "SeqBnConfig",
        "SeqBnInvConfig",
        "StaticAdapterFusionConfig",
        "UniPELTConfig",
    ],
    "context": [
        "AdapterSetup",
        "ForwardContext",
    ],
    "heads": [
        "BertStyleMaskedLMHead",
        "BiaffineParsingHead",
        "CausalLMHead",
        "ClassificationHead",
        "DependencyParsingOutput",
        "ModelWithFlexibleHeadsAdaptersMixin",
        "MultiHeadOutput",
        "MultiLabelClassificationHead",
        "MultipleChoiceHead",
        "PredictionHead",
        "QuestionAnsweringHead",
        "Seq2SeqLMHead",
        "TaggingHead",
    ],
    "interface": ["AdapterMethod", "AdapterModelInterface"],
    "methods.adapter_layer_base": ["AdapterLayerBase", "ComposableAdapterLayerBase"],
    "model_mixin": [
        "EmbeddingAdaptersMixin",
        "InvertibleAdaptersMixin",
        "InvertibleAdaptersWrapperMixin",
        "ModelAdaptersMixin",
        "ModelWithHeadsAdaptersMixin",
    ],
    "models.albert": ["AlbertAdapterModel"],
    "models.auto": [
        "ADAPTER_MODEL_MAPPING",
        "AutoAdapterModel",
    ],
    "models.bart": ["BartAdapterModel"],
    "models.beit": ["BeitAdapterModel"],
    "models.bert": ["BertAdapterModel"],
    "models.bert_generation": ["BertGenerationAdapterModel"],
    "models.clip": ["CLIPAdapterModel"],
    "models.deberta": ["DebertaAdapterModel"],
    "models.deberta_v2": ["DebertaV2AdapterModel"],
    "models.distilbert": ["DistilBertAdapterModel"],
    "models.electra": ["ElectraAdapterModel"],
    "models.gpt2": ["GPT2AdapterModel"],
    "models.gptj": ["GPTJAdapterModel"],
    "models.llama": ["LlamaAdapterModel"],
    "models.mbart": ["MBartAdapterModel"],
    "models.mistral": ["MistralAdapterModel"],
    "models.mt5": ["MT5AdapterModel"],
    "models.plbart": ["PLBartAdapterModel"],
    "models.roberta": ["RobertaAdapterModel"],
    "models.t5": ["T5AdapterModel"],
    "models.vit": ["ViTAdapterModel"],
    "models.whisper": ["WhisperAdapterModel"],
    "models.xlm_roberta": ["XLMRobertaAdapterModel"],
    "models.xmod": ["XmodAdapterModel"],
    "trainer": ["AdapterTrainer", "Seq2SeqAdapterTrainer"],
    "training": [
        "AdapterArguments",
        "setup_adapter_training",
    ],
    "utils": [
        "ADAPTER_CACHE",
        "AdapterInfo",
        "AdapterType",
        "get_adapter_config_hash",
        "get_adapter_info",
        "list_adapters",
    ],
    "wrappers": [
        "init",
        "init_adapters_config",
        "load_model",
    ],
}


if TYPE_CHECKING:
    from .composition import (
        AdapterCompositionBlock,
        BatchSplit,
        Fuse,
        Parallel,
        Split,
        Stack,
        parse_composition,
        validate_composition,
    )
    from .configuration import (
        ADAPTER_CONFIG_MAP,
        ADAPTERFUSION_CONFIG_MAP,
        DEFAULT_ADAPTER_CONFIG,
        DEFAULT_ADAPTERFUSION_CONFIG,
        AdapterConfig,
        AdapterFusionConfig,
        AdapterPlusConfig,
        BnConfig,
        CompacterConfig,
        CompacterPlusPlusConfig,
        ConfigUnion,
        DiReftConfig,
        DoubleSeqBnConfig,
        DoubleSeqBnInvConfig,
        DynamicAdapterFusionConfig,
        IA3Config,
        LoRAConfig,
        LoReftConfig,
        MAMConfig,
        ModelAdaptersConfig,
        NoReftConfig,
        ParBnConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        ReftConfig,
        SeqBnConfig,
        SeqBnInvConfig,
        StaticAdapterFusionConfig,
        UniPELTConfig,
    )
    from .context import AdapterSetup, ForwardContext
    from .heads import (
        BertStyleMaskedLMHead,
        BiaffineParsingHead,
        CausalLMHead,
        ClassificationHead,
        DependencyParsingOutput,
        ModelWithFlexibleHeadsAdaptersMixin,
        MultiHeadOutput,
        MultiLabelClassificationHead,
        MultipleChoiceHead,
        PredictionHead,
        QuestionAnsweringHead,
        Seq2SeqLMHead,
        TaggingHead,
    )
    from .interface import AdapterMethod, AdapterModelInterface
    from .methods.adapter_layer_base import AdapterLayerBase, ComposableAdapterLayerBase
    from .model_mixin import (
        EmbeddingAdaptersMixin,
        InvertibleAdaptersMixin,
        InvertibleAdaptersWrapperMixin,
        ModelAdaptersMixin,
        ModelWithHeadsAdaptersMixin,
    )
    from .models.albert import AlbertAdapterModel
    from .models.auto import ADAPTER_MODEL_MAPPING, AutoAdapterModel
    from .models.bart import BartAdapterModel
    from .models.beit import BeitAdapterModel
    from .models.bert import BertAdapterModel
    from .models.bert_generation import BertGenerationAdapterModel
    from .models.clip import CLIPAdapterModel
    from .models.deberta import DebertaAdapterModel
    from .models.deberta_v2 import DebertaV2AdapterModel
    from .models.distilbert import DistilBertAdapterModel
    from .models.electra import ElectraAdapterModel
    from .models.gpt2 import GPT2AdapterModel
    from .models.gptj import GPTJAdapterModel
    from .models.llama import LlamaAdapterModel
    from .models.mbart import MBartAdapterModel
    from .models.mistral import MistralAdapterModel
    from .models.mt5 import MT5AdapterModel
    from .models.plbart import PLBartAdapterModel
    from .models.roberta import RobertaAdapterModel
    from .models.t5 import T5AdapterModel
    from .models.vit import ViTAdapterModel
    from .models.whisper import WhisperAdapterModel
    from .models.xlm_roberta import XLMRobertaAdapterModel
    from .models.xmod import XmodAdapterModel
    from .trainer import AdapterTrainer, Seq2SeqAdapterTrainer
    from .training import AdapterArguments, setup_adapter_training
    from .utils import (
        ADAPTER_CACHE,
        AdapterInfo,
        AdapterType,
        get_adapter_config_hash,
        get_adapter_info,
        list_adapters,
    )
    from .wrappers import init, init_adapters_config, load_model

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"__version__": __version__},
    )
