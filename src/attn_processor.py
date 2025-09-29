from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention

from src.lora_controller import enable_lora
from diffusers.models.embeddings import apply_rotary_emb


class AttnProcessor(nn.Module):

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cond_hidden_states: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[torch.Tensor] = None,
        use_cond=False,
        sparse_layout=None,
    ) -> torch.FloatTensor:
        vae_conds_len, L_txt, L_img, conds_len = sparse_layout
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        with enable_lora((attn.to_q, attn.to_k, attn.to_v)):
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            with enable_lora((attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)):
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
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if cond_hidden_states is not None:
            cond_query = attn.to_q(cond_hidden_states)
            cond_key = attn.to_k(cond_hidden_states)
            cond_value = attn.to_v(cond_hidden_states)

            cond_query = cond_query.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )
            cond_value = cond_value.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            if attn.norm_q is not None:
                cond_query = attn.norm_q(cond_query)
            if attn.norm_k is not None:
                cond_key = attn.norm_k(cond_key)

        if cond_rotary_emb is not None:
            cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
            cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

        if cond_hidden_states is not None:
            query = torch.cat([query, cond_query], dim=2)
            key = torch.cat([key, cond_key], dim=2)
            value = torch.cat([value, cond_value], dim=2)

        total_len = query.shape[2]
        hidden_encode_len = total_len - conds_len

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            # attn_mask=mask
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            if cond_hidden_states is not None:
                cond_hidden_states = hidden_states[:, L_txt+L_img : L_txt+L_img + conds_len, :]
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :L_txt, :],
                hidden_states[:, L_txt:L_txt+L_img, :],
            )
            with enable_lora((attn.to_out[0],)):
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            with enable_lora((attn.to_add_out,)):
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if cond_hidden_states is not None:
                cond_hidden_states = attn.to_out[0](cond_hidden_states)
                cond_hidden_states = attn.to_out[1](cond_hidden_states)

            # hidden_states = hidden_states[:, :L_img, :]
            return (
                hidden_states,
                encoder_hidden_states,
                cond_hidden_states,
            )
        elif cond_hidden_states is not None:
            # if there are cond_hidden_states, we need to separate the hidden_states and the cond_hidden_states
            cond_hidden_states = hidden_states[
                :, hidden_encode_len : hidden_encode_len + conds_len, :
            ]
            hidden_states = hidden_states[:, :hidden_encode_len, :]
            return hidden_states, cond_hidden_states
        else:
            return hidden_states, None
