from .bottleneck import init_bottleneck
from .invertible import init_invertible_adapters
from .lora import init_lora
from .prompt_tuning import init_prompt_tuning
from .reft import init_reft


METHOD_INIT_MAPPING = {
    "bottleneck": init_bottleneck,
    "lora": init_lora,
    "prompt_tuning": init_prompt_tuning,
    "reft": init_reft,
    "invertible": init_invertible_adapters,
}
