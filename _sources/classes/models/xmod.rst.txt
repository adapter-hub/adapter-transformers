X-MOD
=====

.. important::
    The X-MOD implementation integrated into Transformers already supports adapters.
    To make this implementation compatible with Adapters, a few changes were necessary:

    - Pre-trained X-MOD checkpoints require conversion before they can be used with Adapters. We provide pre-converted checkpoints for the following models:
            - ``facebook/xmod-base`` -> ``AdapterHub/xmod-base`` with languages adapters split into separate repos (e.g. ``AdapterHub/xmod-base-af_ZA``)
    - In Adapters, the X-MOD classes rely on the usual adapter methods instead of the custom methods introduced in Transformers, i.e.:
        - ``set_active_adapters()`` instead of ``set_default_language()``.
        - ``AdapterSetup`` context instead of ``lang_ids`` parameter.

The abstract from the paper is the following:

*Multilingual pre-trained models are known to suffer from the curse of multilinguality, which causes per-language performance to drop as they cover more languages. We address this issue by introducing language-specific modules, which allows us to grow the total capacity of the model, while keeping the total number of trainable parameters per language constant. In contrast with prior work that learns language-specific components post-hoc, we pre-train the modules of our Cross-lingual Modular (X-MOD) models from the start. Our experiments on natural language inference, named entity recognition and question answering show that our approach not only mitigates the negative interference between languages, but also enables positive transfer, resulting in improved monolingual and cross-lingual performance. Furthermore, our approach enables adding languages post-hoc with no measurable drop in performance, no longer limiting the model usage to the set of pre-trained languages.*

XmodAdapterModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.XmodAdapterModel
    :members:
    :inherited-members: XmodPreTrainedModel
