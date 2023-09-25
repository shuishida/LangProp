# LangProp: A General Optimization Framework using Language Models for Code Improvement

## About
LangProp is a framework of generating code using ChatGPT, and evaluate the code performance against a dataset of expected 
outputs in a supervised learning setting. Usually, ChatGPT generates code which is sensible but fails for some edge cases, and then you need to go back and prompt ChatGPT again with the error.
This framework saves you the hassle by automatically feeding in the exceptions back into ChatGPT in a training loop, so that ChatGPT can iteratively improve the code it generates.

The framework works similarly to PyTorch Lightning. You prepare the following:
- a PyTorch-like Dataset that contains the input to the code and the expected output (ground-truth labels), 
- a "model" definition of `setup.txt` and `update.txt` where the setup and update prompts are defined,
- a Trainer which defines the scoring metric, data preprocessing step, etc.
  - The scoring metric can be as simple as the accuracy of the prediction, i.e. `float(result == labels)`.
  - The preprocessing step converts the items retrieved from the dataset into input-output pairs.

## Setup
```bash
conda create -n langprop python=3.7
conda activate langprop
pip install -r ./src/langprop/requirements.txt
```

Set `.env` at the repository root to include the following.
#### If you are using the OpenAI Endpoint
```
export OPENAI_API_TYPE=open_ai
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=<YOUR API KEY>
export OPENAI_MODEL=gpt-3.5-turbo
```

#### If you are using the Azure Endpoint
```
export OPENAI_API_TYPE=azure
export OPENAI_API_BASE=https://eastus.api.cognitive.microsoft.com
export OPENAI_API_KEY=<YOUR API KEY>
export OPENAI_API_VERSION=2023-03-15-preview
export OPENAI_API_ENGINE=gpt_test
```

#### If you want to use a custom LLM
Override the `call_llm` method in the `LangAPI` class in [./src/langprop/lm_api.py](./src/langprop/lm_api.py). 

## Run examples
We have prepared a very simple example of generating code that computes the factorial for a given integer input.
The model definition is in `examples/factorials/model`, the factorial dataset in `examples/factorials/dataset.py`, and the training code in `examples/factorials/run.py`.

At the start, make sure you add `src` to the `PYTHONPATH` by running
```
export PYTHONPATH=./src/:${PYTHONPATH}
```

### Running the setup prompt and getting an initial code generation
```
python ./src/langprop/examples/factorials/test_setup.py
```

You will find that a new `checkpoint.txt` has been generated in the directory `./src/langprop/examples/factorials/`.

### Run a forward pass through the code that was generated in the previous step
```
python ./src/langprop/examples/factorials/test_policy.py
```

### Run a full training loop that involves updates in case there are any exceptions
```
python ./src/langprop/examples/factorials/test_run.py
```

The resulting code (which we call checkpoints) and the log of ChatGPT prompts and queries can be found in `lm_logs` in the root directory.

## Template engine

To automatically generate and parse prompts with new inputs, outputs and exceptions on the fly, we have developed a simple but flexible custom template engine, which can be found at `prompt.py`.
Here are the key features of the template engine.
- Every line that begins with "#" is treated as comments.
- Every line that begins with "$ " or in between "$begin" and "$end" are treated as executable python code, as well as everything inside {{ }}.
- You can define a system prompt, a user prompt and an assistant prompt by inserting a header of `%% SYSTEM`, `%% USER`, or `%% ASSISTANT`.
- You can import another prompt file as `$import path_to_file.txt`, either by giving a relative path or an absolute path.
