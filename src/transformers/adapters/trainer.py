import os
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedModel, Seq2SeqTrainer, Trainer, __version__
from transformers.adapters.composition import AdapterCompositionBlock, Fuse
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.modeling_utils import unwrap_model

from ..configuration_utils import PretrainedConfig
from ..data.data_collator import DataCollator
from ..optimization import Adafactor, AdamW
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..trainer_callback import TrainerCallback, TrainerControl, TrainerState
from ..trainer_pt_utils import get_parameter_names
from ..trainer_utils import EvalPrediction, ShardedDDPOption
from ..training_args import TrainingArguments
from ..utils import CONFIG_NAME, WEIGHTS_NAME, is_sagemaker_mp_enabled, logging


if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


logger = logging.get_logger(__name__)


class AdapterTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=[AdapterTrainerCallback(self)] + callbacks if callbacks else [AdapterTrainerCallback(self)],
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if adapter_names is not None:
            self.model.set_active_adapters(adapter_names)
        # Set the defaults for loading/ saving model & adapters
        if isinstance(self.model, PreTrainedModel):
            model_frozen = getattr(self.model.base_model, "model_frozen", False)
        else:
            model_frozen = False
        if model_frozen and self.model.active_adapters:
            # Check if training AdapterFusion
            self.train_adapter_fusion = (
                isinstance(self.model.active_adapters, Fuse)
                or isinstance(self.model.active_adapters, AdapterCompositionBlock)
                and any([isinstance(child, Fuse) for child in self.model.active_adapters.children])
            )
        if model.active_adapters is None:
            raise ValueError(
                "Expected a model with an active adapter setup."
                "If you want to fully finetune the model use the Trainer class."
            )
        if (self.label_names is None or len(self.label_names) < 1) and model.active_head is not None:
            all_label_names = set()
            for head in model._active_heads:
                all_label_names |= set(model.heads[head].get_label_names())
            self.label_names = list(all_label_names)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if hasattr(self.model, "config") and hasattr(self.model.config, "adapters"):
                match_str = r"adapter_fusion_layer\..*\.value"
                decay_parameters = [name for name in decay_parameters if not re.match(match_str, name)]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_all_adapters(output_dir)
            if self.train_adapter_fusion:
                self.model.save_all_adapter_fusions(output_dir)
            if hasattr(self.model, "heads"):
                self.model.save_all_heads(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint):
        args = self.args
        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            if os.path.isdir(resume_from_checkpoint):
                adapter_loaded = self._load_adapters(resume_from_checkpoint)
                self._load_adapter_fusions(resume_from_checkpoint)
                # Save all heads for a model with heads
                if hasattr(self.model, "heads"):
                    self._load_heads(resume_from_checkpoint)

            if not adapter_loaded:
                raise Exception("Can't find a valid checkpoint at {}".format(resume_from_checkpoint))

    def _load_adapters(self, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "adapter_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_adapter(os.path.join(os.path.join(resume_from_checkpoint, file_name)))
                    adapter_loaded = True
        return adapter_loaded

    def _load_adapter_fusions(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," in file_name:
                    self.model.load_adapter_fusion(os.path.join(resume_from_checkpoint, file_name))

    def _load_heads(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "head_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_head(os.path.join(resume_from_checkpoint, file_name))


class AdapterTrainerCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.pop("model")
        model_frozen = getattr(model.base_model, "model_frozen", False)
        if not model_frozen:
            raise ValueError(
                "The pre-trained model weights are not frozen. For training adapters, please call the train_adapter()"
                " method"
            )

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.pop("model")
        if args.load_best_model_at_end and state.best_model_checkpoint is not None:

            logger.info(f"Loading best adapter(s) from {state.best_model_checkpoint} (score: {state.best_metric}).")
            # attempt to re-load all adapters from checkpoint
            for adapter in model.config.adapters.adapters:
                adapter_dir = os.path.join(state.best_model_checkpoint, adapter)
                if os.path.exists(adapter_dir):
                    model.load_adapter(adapter_dir)
            if self.trainer.train_adapter_fusion:
                logger.info(
                    f"Loading best adapter fusion(s) from {state.best_model_checkpoint} (score: {state.best_metric})."
                )
                # attempt to re-load all adapter fusions from checkpoint
                for fusion in model.config.adapters.fusions:
                    fusion_dir = os.path.join(state.best_model_checkpoint, fusion)
                    if os.path.exists(fusion_dir):
                        model.load_adapter_fusion(fusion_dir)
            model.to(args.device)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # apply adapter fusion weight regularization on the value matrix
        model = kwargs.pop("model")
        if hasattr(self.trainer, 'train_adapter_fusion') and self.trainer.train_adapter_fusion:
            fusion_reg_loss = model.base_model.get_fusion_regularization_loss()
            fusion_reg_loss.backward()


class Seq2SeqAdapterTrainer(AdapterTrainer, Seq2SeqTrainer):
    pass
