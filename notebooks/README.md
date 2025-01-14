# AdapterHub Notebooks

Here you can find a collection of notebooks for AdapterHub and the adapters library.

The table shows notebooks provided by AdapterHub and contained in this folder that show the different possibilities of working with adapters.

As adapters is fully compatible with HuggingFace's Transformers, you can also use the large collection of official and community notebooks there: [🤗 Transformers Notebooks](https://github.com/huggingface/transformers/tree/main/notebooks).

## Chapter 1: The Basics

| Notebook        | Description          |   |
|:----------------|:---------------------|--:|
| [1: Training an Adapter](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb) | How to train a task adapter for a Transformer model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb) |
| [2: Using Adapters for Inference](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/02_Adapter_Inference.ipynb) | How to download and use pre-trained adapters from AdapterHub | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/02_Adapter_Inference.ipynb) |

## Chapter 2: Modularity & Composition

| Notebook        | Description          |   |
|:----------------|:---------------------|--:|
| [3: Adapter Fusion](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/03_Adapter_Fusion.ipynb) | How to combine multiple pre-trained adapters on a new task using `Fuse` composition. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/03_Adapter_Fusion.ipynb) |
| [4: Cross-lingual Transfer](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/04_Cross_Lingual_Transfer.ipynb) | How to perform zero-shot cross-lingual transfer between tasks using the MAD-X setup (`Stack`). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/04_Cross_Lingual_Transfer.ipynb) |
| [5: Parallel Adapter Inference](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/05_Parallel_Adapter_Inference.ipynb) | Using the `Parallel` composition block for inference. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/05_Parallel_Adapter_Inference.ipynb) |
| [6: Adapter merging and Task Arithmetics](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/06_Task_Arithmetics.ipynb) | How to merge multiple adapters to create a new one through Task Arithmetics. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/06_Task_Arithmetics.ipynb) |
| [7: Complex Adapter Configuration](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/07_Complex_Adapter_Configuration.ipynb) | How to flexibly combine multiple adapter methods in complex setups using `ConfigUnion`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/07_Complex_Adapter_Configuration.ipynb) |

## Chapter 3: Additional Notebooks

| Notebook        | Description          |   |
|:----------------|:---------------------|--:|
| [Text Generation](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/Text_Generation_Training.ipynb) | How to train an adapter for language generation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Text_Generation_Training.ipynb) |
| [Training a NER Adapter](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/Adapter_train_NER_with_id2label.ipynb) | How to train an adapter on a named entity recoginition task. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Adapter_train_NER_with_id2label.ipynb) |
| [Adapter Drop Training](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/Adapter_Drop_Training.ipynb) | How to train an adapter using AdapterDrop | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Adapter_Drop_Training.ipynb) |
| [Inference example for id2label](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/Adapter_train_NER_with_id2label.ipynb) | How to use the id2label dictionary for inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Adapter_id2label_inference.ipynb) |
| [NER on Wikiann](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/08_NER_Wikiann.ipynb) | Evaluating adapters on NER on the wikiann dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/08_NER_Wikiann.ipynb) |
| [Finetuning Whisper with Adapters](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/Adapter_Whisper_Audio_FineTuning.ipynb) | Fine Tuning Whisper using LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Adapter_Whisper_Audio_FineTuning.ipynb) |
| [Adapter Training with ReFT](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/ReFT_Adapters_Finetuning.ipynb) | Fine Tuning using ReFT Adapters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/ReFT_Adapters_Finetuning.ipynb) |
| [ViT Fine-Tuning with AdapterPlus](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/ViT_AdapterPlus_FineTuning.ipynb) | ViT Fine-Tuning with AdapterPlus | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/ViT_AdapterPlus_FineTuning.ipynb) |

### Memory Efficient Training
| Notebook        | Description          |   |
|:----------------|:---------------------|--:|
| [QLoRA LLama Finetuning](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/QLoRA_Llama_Finetuning.ipynb) | How to finetune a quantized Llama model for using QLoRA. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/QLoRA_Llama_Finetuning.ipynb) |
| [Gradient Checkpointing](https://github.com/Adapter-Hub/adapters/blob/main/notebooks/Gradient_Checkpointing_Llama.ipynb) | How to finetune a quantized Llama model for using QLoRA. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/QLoRA_Llama_Finetuning.ipynb) |
