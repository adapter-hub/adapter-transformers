import logging
import os
from typing import List, Optional, Union

from transformers.utils.generic import working_or_temp_dir

from .composition import AdapterCompositionBlock


logger = logging.getLogger(__name__)

DEFAULT_TEXT = "<!-- Add some description here -->"
# docstyle-ignore
ADAPTER_CARD_TEMPLATE = """
---
tags:
{tags}
---

# Adapter `{adapter_repo_name}` for {model_name}

An [adapter](https://adapterhub.ml) for the `{model_name}` model that was trained on the {dataset_name} dataset{head_info}.

This adapter was created for usage with the **[Adapters](https://github.com/Adapter-Hub/adapters)** library.

## Usage

First, install `adapters`:

```
pip install -U adapters
```

Now, the adapter can be loaded and activated like this:

```python
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("{model_name}")
adapter_name = model.{load_fn}("{adapter_repo_name}", set_active=True)
```

## Architecture & Training

{architecture_training}

## Evaluation results

{results}

## Citation

{citation}

"""


class PushAdapterToHubMixin:
    """Mixin providing support for uploading adapters to HuggingFace's Model Hub."""

    def _save_adapter_card(
        self,
        save_directory: str,
        adapter_name: str,
        adapter_repo_name: str,
        datasets_tag: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        license: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        load_fn: str = "load_adapter",
        **kwargs,
    ):
        # Key remains "adapter-transformers", see: https://github.com/huggingface/huggingface.js/pull/459
        all_tags = {"adapter-transformers"}
        datasets = set()
        # Dataset/ Task info
        dataset_name = None
        if datasets_tag is not None:
            dataset_name = f"[{datasets_tag}](https://huggingface.co/datasets/{datasets_tag}/)"
            datasets.add(datasets_tag)

        all_tags.add(self.config.model_type)
        if tags is not None:
            all_tags = all_tags | set(tags)
        tag_string = "\n".join([f"- {tag}" for tag in all_tags])
        if datasets:
            tag_string += "\ndatasets:\n"
            tag_string += "\n".join([f"- {tag}" for tag in datasets])
        if language:
            tag_string += f"\nlanguage:\n- {language}"
        if license:
            tag_string += f'\nlicense: "{license}"'
        if metrics:
            tag_string += "\nmetrics:\n"
            tag_string += "\n".join([f"- {metric}" for metric in metrics])

        if hasattr(self, "heads") and adapter_name in self.heads:
            head_type = " ".join(self.heads[adapter_name].config["head_type"].split("_"))
            head_info = f" and includes a prediction head for {head_type}"
        else:
            head_info = ""

        adapter_card = ADAPTER_CARD_TEMPLATE.format(
            tags=tag_string,
            model_name=self.model_name,
            dataset_name=dataset_name,
            head_info=head_info,
            load_fn=load_fn,
            adapter_repo_name=adapter_repo_name,
            architecture_training=kwargs.pop("architecture_training", DEFAULT_TEXT),
            results=kwargs.pop("results", DEFAULT_TEXT),
            citation=kwargs.pop("citation", DEFAULT_TEXT),
        )

        logger.info('Saving adapter card for adapter "%s" to %s.', adapter_name, save_directory)
        with open(os.path.join(save_directory, "README.md"), "w") as f:
            f.write(adapter_card.strip())

    def push_adapter_to_hub(
        self,
        repo_id: str,
        adapter_name: str,
        datasets_tag: Optional[str] = None,
        local_path: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        overwrite_adapter_card: bool = False,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
        adapter_card_kwargs: Optional[dict] = None,
    ):
        """Upload an adapter to HuggingFace's Model Hub.

        Args:
            repo_id (str): The name of the repository on the model hub to upload to.
            adapter_name (str): The name of the adapter to be uploaded.
            datasets_tag (str, optional): Dataset identifier from https://huggingface.co/datasets. Defaults to
                None.
            local_path (str, optional): Local path used as clone directory of the adapter repository.
                If not specified, will create a temporary directory. Defaults to None.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add config"`, :obj:`"add tokenizer"` or
                :obj:`"add model"` depending on the type of the class.
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            overwrite_adapter_card (bool, optional): Overwrite an existing adapter card with a newly generated one.
                If set to `False`, will only generate an adapter card, if none exists. Defaults to False.
            create_pr (bool, optional):
                Whether or not to create a PR with the uploaded files or directly commit.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            adapter_card_kwargs (Optional[dict], optional): Additional arguments to pass to the adapter card text generation.
                Currently includes: tags, language, license, metrics, architecture_training, results, citation.

        Returns:
            str: The url of the adapter repository on the model hub.
        """
        use_temp_dir = not os.path.isdir(local_path) if local_path else True

        # Create repo or get retrieve an existing repo
        repo_id = self._create_repo(repo_id, private=private, token=token)

        # Commit and push
        logger.info('Pushing adapter "%s" to model hub at %s ...', adapter_name, repo_id)
        with working_or_temp_dir(working_dir=local_path, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)
            # Save adapter and optionally create model card
            self.save_adapter(work_dir, adapter_name)
            if overwrite_adapter_card or not os.path.exists(os.path.join(work_dir, "README.md")):
                adapter_card_kwargs = adapter_card_kwargs or {}
                self._save_adapter_card(
                    work_dir,
                    adapter_name,
                    repo_id,
                    datasets_tag=datasets_tag,
                    **adapter_card_kwargs,
                )
            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )

    def push_adapter_setup_to_hub(
        self,
        repo_id: str,
        adapter_setup: Union[str, list, AdapterCompositionBlock],
        head_setup: Optional[Union[bool, str, list, AdapterCompositionBlock]] = None,
        datasets_tag: Optional[str] = None,
        local_path: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        overwrite_adapter_card: bool = False,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
        adapter_card_kwargs: Optional[dict] = None,
    ):
        """Upload an adapter setup to HuggingFace's Model Hub.

        Args:
            repo_id (str): The name of the repository on the model hub to upload to.
            adapter_setup (Union[str, list, AdapterCompositionBlock]): The adapter setup to be uploaded. Usually an adapter composition block.
            head_setup (Optional[Union[bool, str, list, AdapterCompositionBlock]], optional): The head setup to be uploaded.
            datasets_tag (str, optional): Dataset identifier from https://huggingface.co/datasets. Defaults to
                None.
            local_path (str, optional): Local path used as clone directory of the adapter repository.
                If not specified, will create a temporary directory. Defaults to None.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add config"`, :obj:`"add tokenizer"` or
                :obj:`"add model"` depending on the type of the class.
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            overwrite_adapter_card (bool, optional): Overwrite an existing adapter card with a newly generated one.
                If set to `False`, will only generate an adapter card, if none exists. Defaults to False.
            create_pr (bool, optional):
                Whether or not to create a PR with the uploaded files or directly commit.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            adapter_card_kwargs (Optional[dict], optional): Additional arguments to pass to the adapter card text generation.
                Currently includes: tags, language, license, metrics, architecture_training, results, citation.

        Returns:
            str: The url of the adapter repository on the model hub.
        """
        use_temp_dir = not os.path.isdir(local_path) if local_path else True

        # Create repo or get retrieve an existing repo
        repo_id = self._create_repo(repo_id, private=private, token=token)

        # Commit and push
        logger.info('Pushing adapter setup "%s" to model hub at %s ...', adapter_setup, repo_id)
        with working_or_temp_dir(working_dir=local_path, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)
            # Save adapter and optionally create model card
            if head_setup is not None:
                save_kwargs = {"head_setup": head_setup}
            else:
                save_kwargs = {}
            self.save_adapter_setup(work_dir, adapter_setup, **save_kwargs)
            if overwrite_adapter_card or not os.path.exists(os.path.join(work_dir, "README.md")):
                adapter_card_kwargs = adapter_card_kwargs or {}
                self._save_adapter_card(
                    work_dir,
                    str(adapter_setup),
                    repo_id,
                    datasets_tag=datasets_tag,
                    load_fn="load_adapter_setup",
                    **adapter_card_kwargs,
                )
            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )
