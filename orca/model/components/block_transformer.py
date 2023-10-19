# Written by Dibya
from dataclasses import dataclass, replace
import logging

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.transformer import Transformer
from orca.utils.typing import Dict, PRNGKey, Sequence


@dataclass
class PrefixGroup:
    name: str
    tokens: jnp.ndarray  # with shape (batch, n_tokens, token_embedding_size)
    attends_to: Sequence[str]

    def __post_init__(self):
        assert self.tokens.ndim == 3


@dataclass
class TimestepGroup:
    name: str
    tokens: jnp.ndarray  # with shape (batch, horizon, n_tokens, token_embedding_size)
    attends_to: Sequence[str]

    def __post_init__(self):
        assert self.tokens.ndim == 4


@dataclass
class TokenMetadata:
    """Useful metadata for computing attention masks"""

    group_name: str
    timestep: int  # -1 for prefix tokens
    attends_to: Sequence[str]


def split_tokens(ary, n_tokens_per_group, axis):
    cumsum = np.cumsum(n_tokens_per_group)
    return jnp.split(ary, cumsum, axis=axis)


class BlockTransformer(nn.Module):
    num_layers: int = 4
    mlp_dim: int = 1024
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
        timestep_pad_mask: jnp.ndarray,
        train: bool = False,
    ):
        """
        Performs a forward pass of the network with certain computation groups and returns the corresponding embeddings

        Note: By construction, computation groups are independent of one another! The following two calls are equivalent:
        ```
        transformer_embeddings1 = model(observations, tasks,
            computation_groups={"actions": action_tokens})
        transformer_embeddings2 = model(observations, tasks,
            computation_groups={"actions": action_tokens, "value": value_tokens})
        transformer_embeddings1["action"] == transformer_embeddings2["action"]
        ```

        Args:
            observations: A dictionary containing observation data for a batch of trajectory windows.
                Each entry has shape (batch, horizon, *).
            tasks: A dictionary containing task data for the trajectory windows.
                Each entry has shape (batch, *).
            computation_groups: A dictionary {string: transformer_inputs} where transformer_inputs
                has shape (batch, horizon, n_tokens, token_embedding_size)
                (n_tokens may vary between different computation groups)
            train: Whether to use dropout.

        Returns:
            embedding_dict: A dictionary {
                    "task": task_embeddings, # shape (batch, tokens_per_task, token_embedding_size)
                    "obs": obs_embeddings, # shape (batch, horizon, tokens_per_obs, token_embedding_size)
                    **{k: embedding for k in computation_groups} # shape (batch, horizon, computation_groups[k].shape[-2], token_embedding_size)
                }

        Note: Horizon can be anything <= max_horizon.
        """
        logging.warning("Prefix groups:")
        for prefix_group in prefix_groups:
            logging.warning(
                "PrefixGroup(name=%s, shape=%s, attends_to=%s)",
                prefix_group.name,
                prefix_group.tokens.shape,
                prefix_group.attends_to,
            )
        logging.warning("Timestep groups:")
        for timestep_group in timestep_groups:
            logging.warning(
                "TimestepGroup(name=%s, shape=%s, attends_to=%s)",
                timestep_group.name,
                timestep_group.tokens.shape,
                timestep_group.attends_to,
            )

        horizon = timestep_groups[0].tokens.shape[1]
        assert all([group.tokens.shape[1] == horizon for group in timestep_groups])

        attention_mask = self.generate_attention_mask(
            prefix_groups, timestep_groups, timestep_pad_mask
        )
        input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)

        transformer = Transformer(
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_attention_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            add_position_embedding=False,  # we add our own
        )
        output = transformer(input_tokens, attention_mask, train=train)

        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        n_prefix_tokens = sum(tokens_per_prefix_group)

        prefix_embeddings, timestep_embeddings = jnp.split(
            output, [n_prefix_tokens], axis=1
        )
        if len(prefix_groups) > 0:
            prefix_embeddings_split = split_tokens(
                prefix_embeddings, tokens_per_prefix_group, axis=1
            )
            all_prefix_outputs = [
                replace(group, tokens=embeddings)
                for group, embeddings in zip(prefix_groups, prefix_embeddings_split)
            ]
        else:
            all_prefix_outputs = []
        timestep_embeddings = einops.rearrange(
            timestep_embeddings,
            "batch (horizon n_tokens) d -> batch horizon n_tokens d",
            horizon=horizon,
        )

        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]
        timestep_embeddings_split = split_tokens(
            timestep_embeddings, tokens_per_timestep_group, axis=2
        )

        all_timestep_outputs = [
            replace(group, tokens=embeddings)
            for group, embeddings in zip(timestep_groups, timestep_embeddings_split)
        ]
        return all_prefix_outputs, all_timestep_outputs

    def assemble_input_tokens(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
    ):
        """
        - Concatenate all timestep tokens together
        - Fold horizon dim into token sequence dim.
        - Prepend task tokens.
        """
        if len(prefix_groups) > 0:
            all_prefix_tokens = jnp.concatenate(
                [group.tokens for group in prefix_groups], axis=1
            )
        else:
            all_prefix_tokens = jnp.zeros(
                (
                    timestep_groups[0].tokens.shape[0],
                    0,
                    timestep_groups[0].tokens.shape[-1],
                ),
                dtype=jnp.float32,
            )

        all_timestep_tokens = jnp.concatenate(
            [group.tokens for group in timestep_groups], axis=2
        )
        all_timestep_tokens = einops.rearrange(
            all_timestep_tokens,
            "batch horizon n_tokens d -> batch (horizon n_tokens) d",
        )
        tokens = jnp.concatenate([all_prefix_tokens, all_timestep_tokens], axis=1)
        return tokens

    def generate_attention_mask(
        self,
        prefix_groups: Sequence[PrefixGroup],
        timestep_groups: Sequence[TimestepGroup],
        pad_mask: jnp.ndarray,
    ):
        """
        TODO: Need to update this docstring
        Args:
            tokens_per_group: A dictionary {group_name: num_tokens_for_group for group_name in computation_groups}
            horizon: Number of timesteps in the trajectory window.
            pad_mask: A boolean mask of shape (batch, horizon) indicating which timesteps are padding.
        Returns:
            attention_mask: A boolean mask of shape (batch, num_heads, total_tokens, total_tokens)

        Generate default attention mask for transformer call. The default attention mask
        is causal (tokens cannot attend to future tokens) and masks attention to past action
        tokens (since attending to previous actions can hurt performance).

        We generate an NxN mask where the nth row denotes which tokens the nth token
        can attend to.

        attention_mask[i, j] = 1 denotes that token j can attend to token i.
        attention_mask[i, j] = 0 denotes that token j cannot attend to token i.

        This function first creates a lower triangular matrix with past actions masked out.
        Then this causal mask is offset by a non-causal mask for the task tokens.

        For example, given the token sequence: [t_0, t_1, o_0, a_0, o_1, a_1, o_2, a_2]
        the attention mask would be:

        1 1 0 0 0 0 0 0
        1 1 0 0 0 0 0 0
        1 1 0 0 0 0 0 0
        1 1 1 0 0 0 0 0
        1 1 1 0 0 0 0 0
        1 1 1 0 1 0 0 0
        1 1 1 0 1 0 0 0
        1 1 1 0 1 0 1 0
        """

        def _get_position(i, tokens_per_elem):
            return np.searchsorted(np.cumsum(tokens_per_elem), i)

        horizon = pad_mask.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]

        tokens_for_prefix = sum(tokens_per_prefix_group)
        tokens_per_time_step = sum(tokens_per_timestep_group)

        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        attention_mask = np.zeros((total_tokens, total_tokens), dtype=int)

        def get_token_description(i):
            if i < tokens_for_prefix:
                position = _get_position(i, tokens_per_prefix_group)
                group = prefix_groups[position]
                return TokenMetadata(group.name, -1, group.attends_to)
            i -= tokens_for_prefix
            timestep, i = divmod(i, tokens_per_time_step)
            position = _get_position(i, tokens_per_timestep_group)
            group = timestep_groups[position]
            return TokenMetadata(group.name, timestep, group.attends_to)

        for i in range(total_tokens):  # Token attending
            for j in range(total_tokens):  # Token being attended to
                # description is a TokenMetadata(token_name, token_timestep, extra_info)
                description_i = get_token_description(i)
                description_j = get_token_description(j)

                if description_i.group_name == description_j.group_name:
                    mask = int(description_i.timestep <= description_j.timestep)
                elif description_j.group_name in description_i.attends_to:
                    mask = int(description_i.timestep <= description_j.timestep)
                else:
                    mask = 0

                attention_mask[i, j] = mask

        pad_attention_mask = self.generate_pad_attention_mask(
            pad_mask, tokens_per_time_step, tokens_for_prefix
        )
        attention_mask = jnp.logical_and(attention_mask, pad_attention_mask)
        return attention_mask

    def generate_pad_attention_mask(
        self, pad_mask, tokens_per_time_step, tokens_for_prefix
    ):
        """
        Generate attention mask that ignores padding. `pad_mask` has shape (batch, horizon) and
        records which time steps are padding. We first expand the mask to shape (batch, horizon * tokens_per_time_step)
        and then prepend a mask for the task prefix to get shape (batch, total_tokens).
        We broadcast to (batch, num_heads, total_tokens, total_tokens).
        """
        horizon = pad_mask.shape[1]
        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon
        sequence_mask = jnp.repeat(pad_mask, tokens_per_time_step, axis=1)
        task_mask = jnp.ones((pad_mask.shape[0], tokens_for_prefix), dtype=int)
        full_mask = jnp.concatenate([task_mask, sequence_mask], axis=1)
        full_mask = jnp.broadcast_to(
            full_mask[:, None, None, :],
            (
                full_mask.shape[0],
                self.num_attention_heads,
                total_tokens,
                total_tokens,
            ),
        )
        return full_mask
