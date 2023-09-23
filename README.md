# ORCA

This repo contains code for training and finetuning large robot policies.
Currently, ORCA policies are causal transformer models trained on a diverse mix of robot datasets using BC.

![ORCA model](docs/assets/orca_model.jpeg)

We tokenize **task definitions** (like language instructions or goals), **observations** (like RGB-D images and proprioception)
and **actions**. Given the sequence of input tokens, the model is trained to predict the action tokens.

## Installation
```
conda create -n orca python=3.10
conda activate orca
pip install -e .
pip install -r requirements.txt
```
For GPU:
```
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```
pip install --upgrade "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

## Training

### Data
We use the RLDS data format and provide fast, parallelized data loaders for policy training. To download the datasets
please reach out to [pertsch@berkeley.edu](mailto:pertsch@berkeley.edu) or download datasets directly from the
**"Open X-Embodiment" repo.

### Base Policy Training

To train foundational ORCA policies, you can follow the example command below. You can modify hyperparameters like
dataset, batch size etc. in [train_config.py](experiments/main/configs/train_config.py).
```
python experiments/main/train.py --config.data_path=<...> --config.save_dir=<...>
```

### Policy Finetuning

To finetune on of the existing ORCA checkpoints, you can follow the example command below. You can modify hyperparameters like
dataset, batch size etc. in [train_config.py](experiments/main/configs/train_config.py).
```
python experiments/main/train.py --config.data_path=<...> --config.save_dir=<...>
```

## Code Structure

|  | File | Description                                                               |
| --- | --- |---------------------------------------------------------------------------|
| Hyperparameters | [config.py](experiments/main/configs/train_config.py) | Defines all hyperparameters for the training run.                         |
| Training Loop | [train.py](experiments/main/train.py) | Main training script.                                                     |
| Datasets | [dataset.py](orca/data/dataset.py) | Functions for creating single / interleaved datasets + data augmentation. |
| Encoders | [tokenizers.py](orca/model/tokenizers.py) | Tokenizers that encode image / text inputs into tokens.                   |
| Model + Objective | [transformer_policy.py](orca/model/transformer_policy.py) | Sort tokens into sequence, run forward pass, compute loss.                |
| Visualization | [visualization_lib.py](experiments/main/visualization_lib.py) | Utilities for offline qualitative & quantitative eval.                    |

## Contributing
Experimental things and training/eval scripts should go in `experiments/<your_name>`. To make any changes to files outside of your experiments directory, please open a pull request.

To enable code checks and auto-formatting, please install pre-commit hooks:
```
pre-commit install
```

## FAQ

- **Jax complains about wrong CUDA / CuDNN version**: [Jax picks up on the system CuDNN first](https://github.com/google/jax/issues/17497)
(before using the bundled CUDA), so if you encounter version issues please update your system CUDA / CuDNN
or remove it so Jax uses the bundled packages
