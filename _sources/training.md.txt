# Adapter Training

This section describes some examples of training adapter methods for different scenarios. We focus on integrating adapter methods into existing training scripts for Transformer models.
All presented scripts are only slightly modified from the original [examples from Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch#examples).
To run the scripts, make sure you have the latest version of the repository and have installed some additional requirements:

```
git clone https://github.com/adapter-hub/adapters
cd adapters
pip install .
pip install -r ./examples/pytorch/<your_examples_folder>/requirements.txt
```

## Train a Task Adapter

Training a task adapter module on a dataset only requires minor modifications compared to training the entire model.
Suppose we have an existing script for training a Transformer model.
In the following, we will use Hugging Face's [run_glue.py](https://github.com/Adapter-Hub/adapters/blob/main/examples/pytorch/text-classification/run_glue.py) example script for training on the GLUE benchmark.
We go through all required changes step by step:

### Step A - Parse `AdapterArguments`

The [`AdapterArguments`](adapters.training.AdapterArguments) class integrated into adapters provides a set of command-line options useful for training adapters.
These include options such as `--train_adapter` for activating adapter training and `--load_adapter` for loading adapters from checkpoints.
Thus, the first step of integrating adapters is to add these arguments to the line where `HfArgumentParser` is instantiated:

```python
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))
# ...
model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
```

### Step B - Switch model class (optional)

In our example, we replace the built-in `AutoModelForSequenceClassification` class with the `AutoAdapterModel` class introduced by `adapters`.
Therefore, the model instantiation changed to:

```python
model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
)
model.add_classification_head(data_args.task_name, num_labels=num_labels)
```

Alternatively, you can also use the original `transformers` class and initialize the model for the usage of adapters by calling `adapters.init(model)`.
Learn more about the benefits of AdapterModel classes [here](prediction_heads.md)

### Step C - Setup adapter methods

```{eval-rst}
.. tip::
    In the following, we show how to set up adapters manually. In most cases, you can use the built-in ``setup_adapter_training()`` method to perform this job automatically. Just add a statement similar to this anywhere between model instantiation and training start in your script: ``setup_adapter_training(model, adapter_args, task_name)``
```

Compared to fine-tuning the entire model, we have to make only one significant adaptation: adding an adapter setup and activating it.

```python
# task adapter - only add if not existing
if task_name not in model.adapters_config:
    # resolve the adapter config
    adapter_config = AdapterConfig.load(adapter_args.adapter_config)
    # add a new adapter
    model.add_adapter(task_name, config=adapter_config)
# Enable adapter training
model.train_adapter(task_name)
```

```{eval-rst}
.. important::
    The most crucial step when training an adapter module is to freeze all weights in the model except for those of the
    adapter. In the previous snippet, this is achieved by calling the ``train_adapter()`` method, which disables training
    of all weights outside the task adapter. In case you want to unfreeze all model weights later on, you can use
    ``freeze_model(False)``.
```

Besides this, we only have to make sure that the task adapter and prediction head are activated so that they are used in every forward pass. To specify the adapter modules to use, we can use the `model.set_active_adapters()` 
method and pass the adapter setup. If you only use a single adapter, you can simply pass the name of the adapter. For more information
on complex setups, checkout the [Composition Blocks](https://docs.adapterhub.ml/adapter_composition.html).

```python
model.set_active_adapters(task_name)
```

### Step D - Switch to `AdapterTrainer` class

Finally, we exchange the `Trainer` class built into Transformers for the [`AdapterTrainer`](adapters.trainer.AdapterTrainer) class that is optimized for training adapter methods.
See [below for more information](#adaptertrainer).

Technically, this change is not required as no changes to the training loop are required for training adapters.
However, `AdapterTrainer` e.g., provides better support for checkpointing and reloading adapter weights.

### Step E - Start training

The rest of the training procedure does not require any further changes in code.

You can find the full version of the modified training script for GLUE at [run_glue.py](https://github.com/Adapter-Hub/adapters/blob/master/examples/pytorch/text-classification/run_glue.py) in the `examples` folder of our repository.
We also adapted [various other example scripts](https://github.com/Adapter-Hub/adapters/tree/master/examples/pytorch) (e.g., `run_glue.py`, `run_multiple_choice.py`, `run_squad.py`, ...) to support adapter training.

To start adapter training on a GLUE task, you can run something similar to:

```
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config seq_bn
```

The important flag here is `--train_adapter`, which switches from fine-tuning the entire model to training an adapter module for the given GLUE task.

```{eval-rst}
.. tip::
    Adapter weights are usually initialized randomly, which is why we require a higher learning rate. We have found that a default adapter learning rate of ``1e-4`` works well for most settings.
```

```{eval-rst}
.. tip::
    Depending on your data set size, you might also need to train longer than usual. To avoid overfitting, you can evaluate the adapters after each epoch on the development set and only save the best model.
```

## Train a Language Adapter

Training a language adapter is equally straightforward as training a task adapter. Similarly to the steps for task adapters
described above, we add a language adapter module to an existing model training script. Here, we modified Hugging Face's [run_mlm.py](https://github.com/Adapter-Hub/adapters/blob/main/examples/pytorch/language-modeling/run_mlm.py) script for masked language modeling with BERT-based models.

Training a language adapter on BERT using this script may look like the following:

```bash
export TRAIN_FILE=/path/to/dataset/train
export VALIDATION_FILE=/path/to/dataset/validation

python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir /tmp/test-mlm \
    --train_adapter \
    --adapter_config "seq_bn_inv"
```

## Train AdapterFusion

We provide an example for training _AdapterFusion_ ([Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00247)) on the GLUE dataset: [run_fusion_glue.py](https://github.com/Adapter-Hub/adapters/blob/main/examples/pytorch/adapterfusion/run_fusion_glue.py). 
You can adapt this script to train AdapterFusion with different pre-trained adapters on your own dataset.

```{eval-rst}
.. important::
    AdapterFusion on a target task is trained in a second training stage after independently training adapters on individual tasks.
    When setting up a fusion architecture on your model, make sure to load the pre-trained adapter modules to be fused using ``model.load_adapter()`` before adding a fusion layer.
    For more on AdapterFusion, also refer to `Pfeiffer et al., 2020 <https://arxiv.org/pdf/2005.00247>`_.
```

To start fusion training on SST-2 as the target task, you can run something like the following:

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2

python run_fusion_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir
```


## AdapterTrainer

Similar to the `Trainer` class provided by Hugging Face, adapters provides an `AdapterTrainer` class. This class is only
intended for training adapters. The `Trainer` class should still be used to fully fine-tune models. To train adapters with the `AdapterTrainer`
class, simply initialize it the same way you would initialize the `Trainer` class, e.g.: 

```python
from transformers.training_args import TrainingArguments 

model.add_adapter(task_name)
model.train_adapter(task_name)

training_args =  TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
)

trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
```
```{eval-rst}
.. tip::
    When you migrate from the previous versions, which use the Trainer class for adapter training and fully fine-tuning, note that the 
    specialized AdapterTrainer class does not have the parameters `do_save_full_model`, `do_save_adapters` and `do_save_adapter_fusion`.
```

## Quantized Model Training

_Adapters_ supports fine-tuning of quantized language models similar to [QLoRA (Dettmers et al., 2023)](https://arxiv.org/pdf/2305.14314.pdf) via the `bitsandbytes` library integrated into Transformers.
Quantized training is supported for LoRA-based adapters as well as bottleneck adapters and prefix tuning.
Please refer to [this notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/QLoRA_Llama_Finetuning.ipynb) for a hands-on guide.
