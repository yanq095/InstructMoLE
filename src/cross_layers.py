from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb


class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(
        self,
        cond_width=512,
        cond_height=512,
    ):
        super().__init__()
        self.cond_width = cond_width
        self.cond_height = cond_height

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        cross_diff_lambda,
        encoder_hidden_states: torch.FloatTensor = None,
        cond_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, seq_len, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_cond = attn.to_k(cond_hidden_states)
        value_cond = attn.to_v(cond_hidden_states)

        key_cond = key_cond.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_cond = value_cond.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            key_cond = attn.norm_k(key_cond)

        query = apply_rotary_emb(query, image_rotary_emb[0])
        key = apply_rotary_emb(key, image_rotary_emb[0])
        key_cond = apply_rotary_emb(key_cond, image_rotary_emb[1])

        cross_output = F.scaled_dot_product_attention(
            query,
            key_cond,
            value_cond,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = (
            cross_diff_lambda * cross_output
            + F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(
        self,
        cond_width=512,
        cond_height=512,
    ):
        super().__init__()
        self.cond_width = cond_width
        self.cond_height = cond_height

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        cross_diff_lambda,
        cond_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        # `context` projections.
        inner_dim = 3072
        head_dim = inner_dim // attn.heads
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_cond = (
            attn.to_k(cond_hidden_states)
            .view(batch_size, -1, attn.heads, head_dim)
            .transpose(1, 2)
        )
        value_cond = (
            attn.to_v(cond_hidden_states)
            .view(batch_size, -1, attn.heads, head_dim)
            .transpose(1, 2)
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            key_cond = attn.norm_k(key_cond)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb[0])
            key = apply_rotary_emb(key, image_rotary_emb[0])
            key_cond = apply_rotary_emb(key_cond, image_rotary_emb[1])

        cross_output = F.scaled_dot_product_attention(
            query,
            key_cond,
            value_cond,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = (
            cross_diff_lambda * cross_output
            + F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # Linear projection (with LoRA weight applied to each proj layer)
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return (hidden_states, encoder_hidden_states)
