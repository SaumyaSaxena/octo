import functools as ft
import re
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from orca.model.components import encoders
from orca.model.components.clip import CLIPTextTokenizer, CLIPVisionTokenizer
from orca.model.components.transformer import MlpBlock

EPS = 1e-6


# adapted from https://github.com/google-research/robotics_transformer/blob/master/tokenizers/token_learner.py
class TokenLearner(nn.Module):
    """
    Learns to map fixed-length sequence of tokens into specified number of tokens.

    Args:
        num_tokens (int): Number of output tokens.
        bottleneck_dim (int): Size of the hidden layers of the mapping MLP.
        dropout_rate (float): Rate of dropout applied in the mapping MLP. Defaults to no dropout.
    """

    num_tokens: int
    bottleneck_dim: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        input_4_dims = len(inputs.shape) == 4
        if input_4_dims:
            b, t, _, d = inputs.shape
            inputs = inputs.reshape(b * t, -1, d)
        x = nn.LayerNorm()(inputs)
        x = MlpBlock(
            mlp_dim=self.bottleneck_dim,
            out_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
        )(x, deterministic=not train)
        x = jnp.transpose(x, (0, 2, 1))  # (batch, num_tokens, h*w)
        x = nn.softmax(x, axis=-1)
        x = jnp.einsum("bna,baf->bnf", x, inputs)

        if input_4_dims:
            x = x.reshape((b, t, self.num_tokens, d))
        return x


class ImageTokenizer(nn.Module):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning.

    Args:
        encoder (str): Name of used encoder.
        encoder_kwargs (dict, optional): Overwrite dict for encoder hyperparameters.
        early_fusion (bool): If true multi-cam and goal images during first layer of encoder, else fuses after encoding using TokenLearner
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Supports regex.
    """

    encoder: str
    encoder_kwargs: dict = None
    early_fusion: bool = True
    num_tokens: int = 8
    conditioning_type: str = "none"
    obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*")
    task_stack_keys: Sequence[str] = tuple()
    task_film_keys: Sequence[str] = tuple()

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        def extract_inputs(regex_keys, inputs, check_spatial=False):
            extracted_outputs = []
            for r_key in regex_keys:
                for key in filter(re.compile(r_key).match, sorted(inputs.keys())):
                    if check_spatial:
                        assert len(inputs[key].shape) >= 4
                    extracted_outputs.append(inputs[key])
            return extracted_outputs

        # stack all spatial observation and task inputs
        obs_inputs = extract_inputs(
            self.obs_stack_keys, observations, check_spatial=True
        )
        task_inputs = None
        if tasks and self.task_stack_keys:
            task_inputs = extract_inputs(
                self.task_stack_keys, tasks, check_spatial=True
            )
            task_inputs = [t[:, None] for t in task_inputs]

        if self.early_fusion:
            enc_inputs = jnp.concatenate(obs_inputs, axis=-1)
            if task_inputs is not None:
                task_inputs = jnp.concatenate(task_inputs, axis=-1).repeat(
                    enc_inputs.shape[1], axis=1
                )
                enc_inputs = jnp.concatenate([enc_inputs, task_inputs], axis=-1)[
                    :, :, None
                ]
        else:
            # late fusion treats the goal as an extra time-step during CNN encoding
            enc_inputs = jnp.concatenate([o[:, :, None] for o in obs_inputs], axis=2)
            if task_inputs is not None:
                task_inputs = jnp.concatenate(
                    [t[:, :, None] for t in task_inputs], axis=2
                )
                enc_inputs = jnp.concatenate([enc_inputs, task_inputs], axis=1)

        b, t, n, h, w, c = enc_inputs.shape
        enc_inputs = jnp.reshape(enc_inputs, (b * t * n, h, w, c))

        # extract non-spatial FiLM inputs
        encoder_input_kwargs = {}
        if self.task_film_keys:
            film_inputs = jnp.concatenate(
                extract_inputs(self.task_film_keys, tasks), axis=-1
            )
            film_inputs = film_inputs[:, None].repeat(t, axis=1)
            film_inputs = film_inputs[:, :, None].repeat(n, axis=2)
            encoder_input_kwargs.update(
                {"cond_var": jnp.reshape(film_inputs, (b * t * n, -1))}
            )

        # run visual encoder
        image_tokens = encoders[self.encoder](**self.encoder_kwargs)(
            enc_inputs, **encoder_input_kwargs
        )

        d = image_tokens.shape[-1]
        image_tokens = jnp.reshape(image_tokens, (b, t, -1, d))
        if self.early_fusion:
            assert (
                image_tokens.shape[2] == self.num_tokens
            ), f"encoder should produce {self.num_tokens} tokens!"
            return image_tokens

        if task_inputs is not None:
            image_tokens = jnp.reshape(image_tokens, (b, t, n, -1, d))
            obs_tokens, task_tokens = image_tokens[:, :-1], image_tokens[:, -1:]
            image_tokens = jnp.concatenate(
                (obs_tokens, task_tokens.repeat(t - 1, axis=1)), axis=2
            )
            image_tokens = image_tokens.reshape((b, t - 1, -1, d))
        return TokenLearner(num_tokens=self.num_tokens)(image_tokens, train=train)


class LanguageTokenizer(nn.Module):
    """
    Language tokenizer that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

     Args:
         num_tokens (int): Number of output tokens (not enforced).
         encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
         projection_dim (int, optional): Optional output Dense layer projection dimension.
    """

    num_tokens: int = 1
    encoder: str = None
    projection_dim: int = None

    def setup(self):
        if self.projection_dim:
            self.projection = nn.Dense(self.projection_dim, use_bias=False)

        if self.encoder is not None:
            from transformers import AutoConfig, FlaxAutoModel

            config = AutoConfig.from_pretrained(self.encoder)
            self.hf_model = FlaxAutoModel.from_config(config).module

    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        if self.encoder is not None:
            tokens = self.hf_model(**tasks["language_instruction"]).last_hidden_state
            tokens = jax.lax.stop_gradient(tokens)
        else:
            # add a time dimension to language
            if tasks["language_instruction"].ndim == 2:
                tokens = tasks["language_instruction"][:, None, :]
            else:
                tokens = tasks["language_instruction"]

        if self.projection_dim is not None:
            tokens = self.projection(tokens)

        return tokens


class BinTokenizer(nn.Module):
    """
    Tokenizes continuous inputs via dimension-wise binning in given range.

    Args:
        n_bins (int): Number of discrete bins per dimension.
        bin_type (str): Type of binning. ['uniform', 'normal' = Gaussian]
        low (float): Lower bound for bin range.
        high (float): Upper bound for bin range.
    """

    n_bins: int
    bin_type: str = "uniform"
    low: float = 0
    high: float = 1

    def setup(self):
        if self.bin_type == "uniform":
            self.thresholds = jnp.linspace(self.low, self.high, self.n_bins + 1)
        elif self.bin_type == "normal":
            self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.n_bins + 1))
        else:
            raise ValueError(
                f"Binning type {self.bin_type} not supported in BinTokenizer."
            )

    def __call__(self, inputs):
        if self.bin_type == "uniform":
            inputs = jnp.clip(inputs, self.low + EPS, self.high - EPS)
        inputs = inputs[..., None]
        token_one_hot = (inputs < self.thresholds[1:]) & (
            inputs >= self.thresholds[:-1]
        ).astype(jnp.uint8)
        output_tokens = jnp.argmax(token_one_hot, axis=-1)
        return output_tokens

    def decode(self, inputs):
        one_hot = jax.nn.one_hot(inputs, self.n_bins)
        bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
        outputs = jnp.sum(one_hot * bin_avgs, axis=-1)
        return outputs


TOKENIZERS = {
    "image_tokenizer": ImageTokenizer,
    "language_tokenizer": LanguageTokenizer,
    "bin_tokenizer": BinTokenizer,
}


if __name__ == "__main__":
    import numpy as np

    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    action = np.broadcast_to(action, [2, 2, 7])
    tokenizer = BinTokenizer(n_bins=256, bin_type="normal")
    params = tokenizer.init(jax.random.PRNGKey(0), action)
    action_tokens = tokenizer.apply(params, action)
    detokenized_actions = tokenizer.apply(params, action_tokens, method="decode")

    print(action)
    print(action_tokens)
    print(detokenized_actions)
