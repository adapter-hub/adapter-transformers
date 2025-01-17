.PHONY: extra_style_checks quality style test test-adapter-methods test-adapter-models test-examples

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples tests src utils

# this target runs checks on all files

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	python utils/custom_init_isort.py --check_only
	python utils/sort_auto_mappings.py --check_only
	flake8 $(check_dirs)
	python utils/check_inits.py

# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python utils/custom_init_isort.py
	python utils/sort_auto_mappings.py

# this target runs checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
	${MAKE} extra_style_checks

# Library Tests

# run all tests in the library
test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
	python -c "import transformers; print(transformers.__version__)"

# run all tests for the adapter methods for all adapter models
test-adapter-methods:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/

# run a subset of the adapter method tests for all adapter models
# list of all subsets: [core, heads, embeddings, composition, prefix_tuning, prompt_tuning, reft, unipelt, compacter, bottleneck, ia3, lora, config_union]
subset ?=
test-adapter-method-subset:
	@echo "Running subset $(subset)"
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_methods/ -m $(subset)


# run the hugginface test suite for all adapter models
test-adapter-models:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/test_models/

# Run tests for examples
test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/
