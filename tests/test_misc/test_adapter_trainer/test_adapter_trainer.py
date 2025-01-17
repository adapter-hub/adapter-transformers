import os
import unittest
from tempfile import TemporaryDirectory

import torch
from datasets import Dataset

import adapters
from adapters import AutoAdapterModel
from adapters.composition import Fuse, Stack
from adapters.trainer import AdapterTrainer, logger
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.testing_utils import require_bitsandbytes, require_ray, slow, torch_device


class TestAdapterTrainer(unittest.TestCase):
    def get_model_config(self):
        return BertConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=37,
        )

    def test_resume_training(self):

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        with TemporaryDirectory() as tmpdirname:
            model = AutoModelForSequenceClassification.from_config(self.get_model_config())
            adapters.init(model)
            model.add_adapter("adapter")
            model.add_adapter("additional_adapter")
            model.set_active_adapters("adapter")
            model.train_adapter("adapter")

            training_args = TrainingArguments(
                output_dir=tmpdirname,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            # create second model that should resume the training of the first
            model_resume = AutoModelForSequenceClassification.from_config(self.get_model_config())
            adapters.init(model_resume)
            model_resume.add_adapter("adapter")
            model_resume.add_adapter("additional_adapter")
            model_resume.set_active_adapters("adapter")
            model_resume.train_adapter("adapter")
            trainer_resume = AdapterTrainer(
                model=model_resume,
                args=TrainingArguments(do_train=True, max_steps=1, output_dir=tmpdirname),
                train_dataset=train_dataset,
            )
            trainer_resume.train(resume_from_checkpoint=True)

            self.assertEqual(model.adapters_config.adapters, model_resume.adapters_config.adapters)

            for (k1, v1), (k2, v2) in zip(
                trainer.model.state_dict().items(), trainer_resume.model.state_dict().items()
            ):
                self.assertEqual(k1, k2)
                if "adapter" in k1:
                    self.assertTrue(torch.equal(v1, v2), k1)

    def test_resume_training_invalid_checkpoint(self):

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        with TemporaryDirectory() as tmpdirname:
            model = AutoModelForSequenceClassification.from_config(self.get_model_config())
            adapters.init(model)
            model.add_adapter("adapter")
            model.add_adapter("additional_adapter")
            model.set_active_adapters("adapter")
            model.train_adapter("adapter")

            training_args = TrainingArguments(
                output_dir=tmpdirname,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            # create second model that should resume the training of the first
            model_resume = AutoModelForSequenceClassification.from_config(self.get_model_config())
            adapters.init(model_resume)
            model_resume.add_adapter("adapter")
            model_resume.add_adapter("additional_adapter")
            model_resume.set_active_adapters("adapter")
            model_resume.train_adapter("adapter")
            trainer_resume = AdapterTrainer(
                model=model_resume,
                args=TrainingArguments(do_train=True, max_steps=1, output_dir=tmpdirname),
                train_dataset=train_dataset,
            )
            with self.assertRaises(Exception):
                trainer_resume.train(resume_from_checkpoint=tmpdirname + "_invalid")

    def test_resume_training_with_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        with TemporaryDirectory() as tmpdirname:
            model = AutoModelForSequenceClassification.from_config(self.get_model_config())
            adapters.init(model)
            model.add_adapter("adapter")
            model.add_adapter("additional_adapter")
            model.add_adapter_fusion(Fuse("adapter", "additional_adapter"))
            model.set_active_adapters(Fuse("adapter", "additional_adapter"))
            model.train_adapter_fusion(Fuse("adapter", "additional_adapter"))

            training_args = TrainingArguments(
                output_dir=tmpdirname,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            model_resume = AutoModelForSequenceClassification.from_config(self.get_model_config())
            adapters.init(model_resume)
            model_resume.add_adapter("adapter")
            model_resume.add_adapter("additional_adapter")
            model_resume.add_adapter_fusion(Fuse("adapter", "additional_adapter"))
            model_resume.set_active_adapters(Fuse("adapter", "additional_adapter"))
            model_resume.train_adapter_fusion(Fuse("adapter", "additional_adapter"))
            trainer_resume = AdapterTrainer(
                model=model_resume,
                args=TrainingArguments(do_train=True, max_steps=1, output_dir=tmpdirname),
                train_dataset=train_dataset,
            )
            trainer_resume.train(resume_from_checkpoint=True)

            self.assertEqual(model.adapters_config.adapters, model_resume.adapters_config.adapters)

            for (k1, v1), (k2, v2) in zip(
                trainer.model.to("cpu").state_dict().items(), trainer_resume.model.to("cpu").state_dict().items()
            ):
                self.assertEqual(k1, k2)
                if "adapter" in k1:
                    self.assertTrue(torch.equal(v1, v2), k1)

    def test_auto_set_save_adapters(self):
        model = BertForSequenceClassification(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            )
        )
        adapters.init(model)
        model.add_adapter("adapter1")
        model.add_adapter("adapter2")
        model.add_adapter_fusion(Fuse("adapter1", "adapter2"))
        model.train_adapter_fusion(Fuse("adapter1", "adapter2"))

        with TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
            )
            self.assertTrue(trainer.train_adapter_fusion)

    @slow
    def test_training_load_best_model_at_end_full_model(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_config(self.get_model_config())
        adapters.init(model)
        model.add_adapter("adapter")
        model.train_adapter("adapter")

        with TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,
                do_train=True,
                learning_rate=0.001,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=2,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            trainer.train()
            self.assertIsNotNone(trainer.model.active_adapters)

    def test_training_load_best_model_at_end_adapter(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_config(self.get_model_config())
        adapters.init(model)
        model.add_adapter("adapter")
        model.train_adapter("adapter")

        with TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,
                do_train=True,
                learning_rate=0.001,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=2,
            )
            trainer = AdapterTrainer(
                model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
            )
            with self.assertLogs(logger) as cm:
                trainer.train()
                self.assertTrue(any("Loading best adapter(s) from" in line for line in cm.output))
            self.assertEqual(Stack("adapter"), trainer.model.active_adapters)

    def test_training_load_best_model_at_end_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        model = AutoModelForSequenceClassification.from_config(self.get_model_config())
        adapters.init(model)
        model.add_adapter("fuse_adapter_1")
        model.add_adapter("fuse_adapter_2")
        model.add_adapter_fusion(Fuse("fuse_adapter_1", "fuse_adapter_2"))
        model.train_adapter_fusion(Fuse("fuse_adapter_1", "fuse_adapter_2"))

        with TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,
                do_train=True,
                learning_rate=0.001,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=2,
            )
            trainer = AdapterTrainer(
                model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
            )
            with self.assertLogs(logger) as cm:
                trainer.train()
                self.assertTrue(any("Loading best adapter fusion(s) from" in line for line in cm.output))
            self.assertEqual(Fuse("fuse_adapter_1", "fuse_adapter_2"), trainer.model.active_adapters)

    def test_reloading_prediction_head(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoAdapterModel.from_config(self.get_model_config())

        model.add_classification_head("adapter", num_labels=3)
        model.add_classification_head("dummy", num_labels=2)

        # add the adapters to be fused
        model.add_adapter("adapter")
        model.add_adapter("additional_adapter")

        # setup fusion
        adapter_setup = Fuse("adapter", "additional_adapter")
        model.add_adapter_fusion(adapter_setup)
        model.train_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        self.assertEqual(adapter_setup, model.active_adapters)
        self.assertEqual("dummy", model.active_head)
        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()
            # create second model that should resume the training of the first
            model_resume = AutoAdapterModel.from_config(self.get_model_config())

            model_resume.add_classification_head("adapter", num_labels=3)
            model_resume.add_classification_head("dummy", num_labels=2)
            model_resume.add_adapter("adapter")
            model_resume.add_adapter("additional_adapter")
            # setup fusion
            adapter_setup = Fuse("adapter", "additional_adapter")
            model_resume.add_adapter_fusion(adapter_setup)
            model_resume.train_adapter_fusion(adapter_setup)
            model_resume.set_active_adapters(adapter_setup)
            trainer_resume = AdapterTrainer(
                model=model_resume,
                args=TrainingArguments(do_train=True, max_steps=1, output_dir=tempdir),
                train_dataset=train_dataset,
            )
            trainer_resume.train(resume_from_checkpoint=True)

            self.assertEqual("dummy", model.active_head)
            self.assertEqual(model.adapters_config.adapters, model_resume.adapters_config.adapters)

            for (k1, v1), (k2, v2) in zip(
                trainer.model.to("cpu").state_dict().items(), trainer_resume.model.to("cpu").state_dict().items()
            ):
                self.assertEqual(k1, k2)
                if "adapter" in k1 or "dummy" in k1:
                    self.assertTrue(torch.equal(v1, v2), k1)

    def test_general(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoAdapterModel.from_config(self.get_model_config())

        model.add_classification_head("task", num_labels=3)

        # add the adapters to be fused
        model.add_adapter("task")
        model.add_adapter("additional_adapter")

        model.train_adapter("task")
        self.assertEqual("task", model.active_head)
        self.assertEqual(Stack("task"), model.active_adapters)
        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()

            # Check that adapters are actually saved but the full model is not
            files_dir_checkpoint = [file_or_dir for file_or_dir in os.listdir(os.path.join(tempdir, "checkpoint-1"))]
            self.assertTrue("task" in files_dir_checkpoint)
            self.assertTrue("additional_adapter" in files_dir_checkpoint)
            # Check that full model weights are not stored
            self.assertFalse("pytorch_model.bin" in files_dir_checkpoint)

            # this should always be false in the adapter trainer
            self.assertFalse(trainer.args.remove_unused_columns)
            self.assertEqual("task", model.active_head)
            self.assertEqual(Stack("task"), model.active_adapters)

    def test_train_with_frozen_adapter_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")

        model = AutoAdapterModel.from_config(self.get_model_config())

        model.add_adapter("a")
        model.add_adapter("b")

        adapter_setup = Fuse("a", "b")

        model.add_adapter_fusion(adapter_setup, set_active=True)

        model.add_adapter("c")
        model.add_classification_head("c")

        model.train_adapter("c")

        model.active_adapters = Stack(Fuse("a", "b"), "c")

        # Since our config has a value matrix, make sure it is regularized.
        # We do this by patching the fusion regularization function.
        regularization_called = False
        orig_fusion_regularization_loss = model.base_model.get_fusion_regularization_loss

        def patched_fusion_reg_loss():
            nonlocal regularization_called
            regularization_called = True
            return orig_fusion_regularization_loss()

        model.base_model.get_fusion_regularization_loss = patched_fusion_reg_loss

        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()

        self.assertTrue(regularization_called)

    @require_ray
    def test_hyperparameter_search_works_with_AdapterTrainer(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./hf_transformers/tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        eval_dataset = train_dataset

        def hp_space(params):
            from ray import tune

            return {
                "learning_rate": tune.choice([0.1, 0.2]),
            }

        def model_init(trail=None):
            model = AutoAdapterModel.from_config(self.get_model_config())

            model.add_classification_head("task", num_labels=3)

            # add the adapters to be fused
            model.add_adapter("task")
            model.add_adapter("additional_adapter")

            model.train_adapter("task")
            return model

        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                do_train=True,
                learning_rate=0.1,
                logging_steps=1,
                max_steps=1,
                save_steps=1,
                remove_unused_columns=False,
            )
            trainer = AdapterTrainer(
                model=None,
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, backend="ray", n_trials=2)

    @parameterized.expand(["lora", "seq_bn"])
    @require_bitsandbytes
    def test_quantized_training(self, config):
        model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = Dataset.from_dict({"text": ["Hello, I'm a single sentence!", "This is another sentence."]})

        def tokenize(element):
            return tokenizer(
                element["text"],
                truncation=True,
                max_length=512,  # can set to longer values such as 2048
                add_special_tokens=False,
            )

        dataset_tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

        adapters.init(model)
        model.add_adapter("task", config=config)
        model.train_adapter("task")

        model.adapter_to("task", device=torch_device)

        for param in model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)

        self.assertEqual(Stack("task"), model.active_adapters)
        with TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                evaluation_strategy="steps",
                logging_steps=10,
                max_steps=5,
                lr_scheduler_type="constant",
                optim="paged_adamw_32bit",
                learning_rate=0.0002,
                group_by_length=True,
                bf16=True,
                max_grad_norm=0.3,
            )
            trainer = AdapterTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                train_dataset=dataset_tokenized,
                args=training_args,
            )

            trainer.train()


if __name__ == "__main__":
    unittest.main()
