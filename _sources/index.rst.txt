.. adapters documentation main file, created by
   sphinx-quickstart on Sat Apr 18 10:21:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AdapterHub Documentation
================================================

.. note::
   This documentation is based on the new *Adapters* library.

   The documentation based on the legacy *adapter-transformers* library can be found at: `https://docs-legacy.adapterhub.ml <https://docs-legacy.adapterhub.ml>`_.

*AdapterHub* is a framework simplifying the integration, training and usage of adapters and other efficient fine-tuning methods for Transformer-based language models.
For a full list of currently implemented methods, see the `table in our repository <https://github.com/adapter-hub/adapters#implemented-methods>`_.

The framework consists of two main components:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - `Adapters <https://github.com/adapter-hub/adapters>`_
     - `AdapterHub.ml <https://adapterhub.ml/explore>`_
   * - an add-on to Hugging Face's `Transformers <https://huggingface.co/transformers/>`_ library that adds adapters into transformer models
     - a central collection of pre-trained adapter modules

Currently, we support the PyTorch versions of all models as listed on the `Model Overview <model_overview.html>`_ page.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   training
   transitioning

.. toctree::
   :maxdepth: 2
   :caption: Adapter Methods

   overview
   methods
   method_combinations

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   adapter_composition
   merging_adapters
   prediction_heads
   embeddings
   extending

.. toctree::
   :maxdepth: 2
   :caption: Loading and Sharing

   loading
   huggingface_hub

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   model_overview
   classes/models/albert
   classes/models/auto
   classes/models/bart
   classes/models/beit
   classes/models/bert
   classes/models/bert-generation
   classes/models/clip
   classes/models/deberta
   classes/models/deberta_v2
   classes/models/distilbert
   classes/models/electra
   classes/models/encoderdecoder
   classes/models/gpt2
   classes/models/gptj
   classes/models/llama
   classes/models/mistral
   classes/models/mbart
   classes/models/mt5
   classes/models/plbart
   classes/models/roberta
   classes/models/t5
   classes/models/vit
   classes/models/whisper
   classes/models/xlmroberta
   classes/models/xmod

.. toctree::
   :maxdepth: 1
   :caption: Adapter-Related Classes

   classes/adapter_config
   classes/model_adapters_config
   classes/adapter_layer
   classes/model_mixins
   classes/adapter_training
   classes/adapter_utils

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing
   contributing/adding_adapter_methods
   contributing/adding_adapters_to_a_model

Citation
========

If you use _Adapters_ in your work, please consider citing our library paper `Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning <https://arxiv.org/abs/2311.11077)>`


.. code-block:: bibtex

   @inproceedings{poth-etal-2023-adapters,
      title = "Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning",
      author = {Poth, Clifton  and
         Sterz, Hannah  and
         Paul, Indraneil  and
         Purkayastha, Sukannya  and
         Engl{\"a}nder, Leon  and
         Imhof, Timo  and
         Vuli{\'c}, Ivan  and
         Ruder, Sebastian  and
         Gurevych, Iryna  and
         Pfeiffer, Jonas},
      booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
      month = dec,
      year = "2023",
      address = "Singapore",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2023.emnlp-demo.13",
      pages = "149--160",
   }


Alternatively, for the predecessor `adapter-transformers`, the Hub infrastructure and adapters uploaded by the AdapterHub team, please consider citing our initial paper: `AdapterHub: A Framework for Adapting Transformers <https://arxiv.org/abs/2007.07779>`_


.. code-block:: bibtex

   @inproceedings{pfeiffer2020AdapterHub,
      title={AdapterHub: A Framework for Adapting Transformers},
      author={Jonas Pfeiffer and
               Andreas R\"uckl\'{e} and
               Clifton Poth and
               Aishwarya Kamath and
               Ivan Vuli\'{c} and
               Sebastian Ruder and
               Kyunghyun Cho and
               Iryna Gurevych},
      booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020): Systems Demonstrations},
      year={2020},
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/2020.emnlp-demos.7",
      pages = "46--54",
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
