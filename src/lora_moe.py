# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.nn.init as init
import os

def compute_orthogonality_loss(expert_outputs: torch.Tensor) -> torch.Tensor:
    """
    计算专家输出的正交性损失。

    Args:
        expert_outputs (torch.Tensor): 所有专家的输出，形状为 [Num_Experts, B*S, Dim_Out]

    Returns:
        torch.Tensor: 一个标量损失值。
    """
    num_experts, _, dim_out = expert_outputs.shape
    
    # 1. 归一化每个专家的输出 (这是计算余弦相似度的标准步骤)
    #    [E, N, D] -> [E, N*D]
    outputs_flat = expert_outputs.view(num_experts, -1)
    norm_outputs = F.normalize(outputs_flat, p=2, dim=1)
    
    # 2. 计算成对余弦相似度矩阵
    #    (E, N*D) x (N*D, E) -> (E, E)
    similarity_matrix = torch.matmul(norm_outputs, norm_outputs.t())
    
    # 3. 构建损失
    # 我们只想惩罚不同专家之间的相似度 (非对角线元素)
    # 使用 torch.ones_like 创建一个对角线为0，其余为1的掩码
    mask = 1 - torch.eye(num_experts, device=similarity_matrix.device)
    
    # 将损失定义为非对角线元素的平方的均值，这会更强地惩罚高相似度
    loss = (similarity_matrix * mask).pow(2).sum() / (num_experts * (num_experts - 1))
    
    return loss

# ===============================================================================
# 1. 第一个 Gate 类: 用于 Token-Based 的 Top-k 硬路由
# ===============================================================================
class TopKGate(nn.Module):
    """
    一个用于MoE的Top-K门控网络 (权重未归一化版本)。
    返回的weights是原始的top-k softmax分数，它们的和小于1。
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        aux_loss_alpha: float = 0.01
    ):
        super().__init__()
        if num_experts_per_tok > num_experts:
            raise ValueError("num_experts_per_tok必须小于或等于num_experts。")
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.alpha = aux_loss_alpha
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.embed_dim)))
        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, aux_loss_alpha={self.alpha}, weight_normalized=False"
        )
    def forward(self, hidden_states: torch.Tensor):
        # 1. 计算路由Logits
        hidden_states_flat = hidden_states.view(-1, self.embed_dim)
        logits = F.linear(hidden_states_flat, self.weight)
        # 2. 注入噪声以促进负载均衡 (仅在训练时)
        if self.training:
            noise = torch.randn_like(logits) * (1.0 / self.num_experts)
            logits += noise
        # 3. 计算门控分数 (Softmax)
        scores = logits.softmax(dim=-1)
        # 4. 选择Top-K专家
        # `topk`返回 (values, indices)，这里分别是分数和专家的索引
        # 在这个版本中，我们直接使用这些原始分数作为权重
        weights, indices = torch.topk(scores, k=self.top_k, dim=-1)
        # 5. 计算辅助损失 (仅在训练时)
        aux_loss = None
        if self.training and self.alpha > 0.0:
            expert_mask = F.one_hot(indices, num_classes=self.num_experts).sum(dim=1).float()
            fraction_of_tokens_per_expert = expert_mask.mean(dim=0)
            avg_routing_prob_per_expert = scores.mean(dim=0)
            aux_loss = self.num_experts * (fraction_of_tokens_per_expert * avg_routing_prob_per_expert).sum()
            aux_loss *= self.alpha
        return indices, weights, aux_loss

# ===============================================================================
# 2. 第二个 Gate 类: 用于 Type-Based 的软路由
#    (包含可学习温度，只输出 scores)
# ===============================================================================
class SoftGate(nn.Module):
    """
    A Gate for soft routing, typically used for type-based MoE.
    It features a learnable temperature and returns a full softmax distribution.
    """
    def __init__(self, embed_dim: int, num_experts: int = 16, initial_temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.empty((self.num_experts, self.embed_dim)))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def extra_repr(self) -> str:
        temp = torch.exp(self.log_temperature).item()
        return f"embed_dim={self.embed_dim}, num_experts={self.num_experts}, routing_mode=SOFT, learnable_temp={temp:.4f}"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.numel() // self.embed_dim
        hidden_states_flat = hidden_states.view(num_tokens, self.embed_dim)
        logits = F.linear(hidden_states_flat, self.weight)
        
        temperature = torch.exp(self.log_temperature).clamp(min=0.01)
        scores = (logits / temperature).softmax(dim=-1)
        
        return scores

# ===============================================================================
# 3. 第三个 Gate 类: 用于 Token-Based 的 Expert-Race 硬路由
# ===============================================================================
class ExpertRaceGate(nn.Module):
    """
    Implements the Expert Race gating mechanism for Token-Based routing.
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        num_experts_per_tok: float = 2.0, # 'k' in the paper
        momentum: float = 0.99,
        router_similarity_weight: float = 1e-5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.avg_experts_per_tok = num_experts_per_tok
        self.momentum = momentum
        self.router_similarity_weight = router_similarity_weight

        self.router = nn.Linear(embed_dim, self.num_experts, bias=False)
        self.register_buffer("tau", torch.tensor(0.0, dtype=torch.float32))
        
        # Attribute to store the latest mask for debugging
        self.latest_mask = None
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
        self.tau.fill_(0.0)

    def _compute_lsim(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_tokens, num_experts = logits.shape  # N, E
        if num_experts <= 1:
            return torch.tensor(0.0, device=logits.device)

        M = mask.to(logits.dtype)  # Indicator matrix (N, E)
        P = F.softmax(logits, dim=-1)  # Shape: (N, E)
        P_prime = P.T @ P  # Shape: (E, E)
        M_prime = M.T @ M  # Shape: (E, E)
        eps = torch.finfo(M_prime.dtype).eps
        M_prime_diag = torch.diag(M_prime)
        sum_M_prime_diag = M_prime_diag.sum() + eps
        sum_M_prime_all = M_prime.sum() + eps
        W = torch.zeros_like(M_prime)
        off_diag_factor = (num_experts * num_experts - num_experts) / sum_M_prime_all
        W = M_prime * off_diag_factor
        diag_factor = num_experts / sum_M_prime_diag
        W.fill_diagonal_(0)
        W = W + torch.diag_embed(M_prime_diag * diag_factor)
        lsim = torch.sum(W * P_prime) / num_experts
        return lsim.to(logits.dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            hidden_states (torch.Tensor): Input of shape [N, D] (num_tokens, dim).

        Returns:
            A tuple containing:
            - final_weights (torch.Tensor): A sparse weight matrix of shape [N, E].
            - lsim (torch.Tensor | None): The router similarity loss scalar.
        """
        num_tokens, dim = hidden_states.shape
        logits = self.router(hidden_states) # Shape: [N, E]
        
        lsim = None
        if self.training:
            K = int(num_tokens * self.avg_experts_per_tok)
            K = max(1, min(K, num_tokens * self.num_experts))
            
            score_flat = logits.view(-1)
            kth_val = -torch.kthvalue(-score_flat, k=K).values
            
            self.tau.mul_(self.momentum).add_(kth_val.detach() * (1.0 - self.momentum))
            
            mask = logits >= kth_val
            lsim = self._compute_lsim(logits, mask)
        else:
            mask = logits >= self.tau
            
        final_weights = logits * mask
        self.latest_mask = mask.detach()
        
        return final_weights, lsim
    
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: int = 4,
    ):
        super().__init__()
        assert rank > 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.left = nn.Parameter(torch.zeros(in_features, rank))
        self.right = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but should not affect performance
        nn.init.kaiming_uniform_(self.left, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.right, a=math.sqrt(5))
        nn.init.zeros_(self.right)

    def forward(self, x: torch.Tensor):
        return x @ self.left @ self.right * self.scaling

    def extra_repr(self) -> str:
        # print layer info
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, scaling={self.scaling}"


class LoRAMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    type_embedding = None
    debug = False
    aux_loss_weight = 1

    @classmethod
    def set_type_embedding(cls, type_embedding):
        cls.type_embedding = type_embedding
    
    @classmethod
    def set_aux_loss_weight(cls, aux_loss_weight):
        cls.aux_loss_weight = aux_loss_weight

    def __init__(
        self,
        linear_layer,
        num_experts=8,
        num_experts_per_tok=2,
        rank=4,
        alpha=4,
        train_mole_only=True,
        route_by_type=False,
        integrate_pretrained_expert=False,  # True
        type_aux_loss_alpha=0.1,  # 0
        token_aux_loss_alpha=0.01,  # 0
        orthogonal_reg_alpha=0.1,  # 0
    ):
        super().__init__()
        self.without_experts = False
        self.linear_layer = linear_layer
        if orthogonal_reg_alpha == 0.0:
            self.orthogonal_reg_alpha = None
        else:
            self.orthogonal_reg_alpha = orthogonal_reg_alpha
        if train_mole_only:
            for param in self.linear_layer.parameters():
                param.requires_grad = False
        self.num_experts_per_tok = num_experts_per_tok
        self.integrate_pretrained_expert = integrate_pretrained_expert
        experts = [
            LowRankLinear(
                in_features=self.linear_layer.in_features,
                out_features=self.linear_layer.out_features,
                rank=rank,
                alpha=alpha,
            )
            for i in range(num_experts)
        ]
        if self.integrate_pretrained_expert:
            experts = [linear_layer] + experts
        self.experts = nn.ModuleList(experts)
        self.route_by_type = route_by_type
        if route_by_type:
            # self.gate = SoftGate(
            #     embed_dim=768, 
            #     num_experts=num_experts,
            #     initial_temperature=1.0,
            # )
            # self.forward = self.forward_by_type
            # print(f"LoRAMoE Info: Using Type-Based SOFT Routing.")
            self.gate = TopKGate(
                embed_dim=768, 
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                aux_loss_alpha=type_aux_loss_alpha,
            )
            self.forward = self.forward_by_type_topk
            print(f"LoRAMoE Info: Type-Based HARD Routing.")
        else:
            self.gate = TopKGate(
                embed_dim=self.linear_layer.in_features,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                aux_loss_alpha=token_aux_loss_alpha,
            )
            self.forward = self.forward_by_token

            # self.gate = ExpertRaceGate(
            #         embed_dim=self.linear_layer.in_features,
            #         num_experts=num_experts,
            #         num_experts_per_tok=num_experts_per_tok,
            #     )
            # self.forward = self.forward_token_expert_race
            # print(f"LoRAMoE Info: Using Token-Based EXPERT-RACE Routing.")
            
            # EC-DiT Router (Weight W_r)
            # self.num_experts = num_experts
            # self.embed_dim = self.linear_layer.in_features
            # self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False)
            # self.capacity_factor = num_experts_per_tok
            # # --- Initialization ---
            # init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
            # self.forward = self.forward_by_ec

            print(f"LoRAMoE Info: Token-Based HARD (top-{num_experts_per_tok}) Routing.")

    @torch.no_grad()
    def ec_infer(
        self, hidden_states: torch.Tensor, top_k_indices, top_k_scores
    ) -> torch.Tensor:
        # --- 步骤 1: 获取正确的输入和输出维度 ---
        bsz, seq_len, in_dim = hidden_states.shape
        # 从线性层获取正确的输出维度
        out_dim = self.linear_layer.out_features

        # --- 步骤 2: 使用正确的输出维度初始化 y (Bug修复) ---
        # 原始代码: y = torch.zeros_like(hidden_states)
        y = torch.zeros(bsz, seq_len, out_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        # Loop through experts
        for i in range(self.num_experts):
            expert_token_indices = top_k_indices[:, i, :]  # Shape: (B, C)
            expert_gating_scores = top_k_scores[:, i, :]  # Shape: (B, C)

            # --- 步骤 3: 为 gather 创建与输入维度匹配的索引 ---
            # 这个索引用于从 `hidden_states` (D_in) 中 gather 数据，所以它的维度是 in_dim
            indices_for_gather = expert_token_indices.unsqueeze(-1).expand(
                -1, -1, in_dim
            )
            # Gather along sequence dim (dim=1)
            gathered_tokens = hidden_states.gather(
                dim=1, index=indices_for_gather
            )  # Shape: (B, C, D_in)

            # Compute expert output
            expert_output = self.experts[i](gathered_tokens)  # Shape: (B, C, D_out)

            # Weight the output
            weighted_output = expert_output * expert_gating_scores.unsqueeze(
                -1
            )  # Shape: (B, C, D_out)

            # --- 步骤 4: 为 scatter_add_ 创建与输出维度匹配的索引 (Bug修复) ---
            # 这个索引用于将 `weighted_output` (D_out) scatter 回 `y` (D_out)
            # 它的维度必须与 `weighted_output` 的维度匹配
            indices_for_scatter = expert_token_indices.unsqueeze(-1).expand(
                -1, -1, out_dim
            )

            # Scatter add results back to the output tensor y
            # Use scatter_add_ along sequence dimension (dim=1)
            y.scatter_add_(
                dim=1, index=indices_for_scatter, src=weighted_output.to(y.dtype)
            )
        return y

    def forward_by_ec(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Expert-Choice routing. Uses einsum-based logic
        during training and sparse gather/scatter during inference.

        Args:
            hidden_states (torch.Tensor): Input tensor (B, S, D).

        Returns:
            torch.Tensor: Output tensor after MoE and residual (B, S, D).
        """
        if self.without_experts:
            self.without_experts = False
            return self.linear_layer(hidden_states)
        identity = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        assert dim == self.embed_dim, "Input dimension mismatch"
        # Ensure router/experts are on the correct device
        self.gate.to(hidden_states.device)
        for expert in self.experts:
            expert.to(hidden_states.device)

        # 1. Compute token-expert affinity scores (Eq. 5)
        # Input x' in paper = hidden_states + MHCA(xs) -> Using just hidden_states here
        # as MHCA part isn't available directly. If needed, modify input.
        router_input = hidden_states  # Or x' if available
        logits = self.gate(router_input)  # Shape: (B, S, E)
        # Softmax along expert dim (dim=-1) for each token
        affinity_scores = F.softmax(logits, dim=-1)  # Shape: (B, S, E)

        # Transpose for expert-choice selection (each expert looks at all tokens)
        affinity_T = affinity_scores.permute(0, 2, 1)  # Shape: (B, E, S)

        # 2. Select top-C tokens for each expert (Eq. 6 / Alg 1, step 2)
        # Calculate capacity C = ceil(S * fc / E)
        expert_capacity = math.ceil(seq_len * self.capacity_factor / self.num_experts)
        expert_capacity = min(expert_capacity, seq_len)  # Cannot exceed sequence length
        # print(f"Expert capacity C = {expert_capacity}") # Debug

        # Get top C affinity scores and their indices (relative to S dim) for each expert
        # topk along the last dim (S)
        top_k_scores, top_k_indices = torch.topk(
            affinity_T, k=expert_capacity, dim=-1
        )  # Shapes: (B, E, C)

        if self.training:
            # --- Training Path: Einsum-based (potentially slower) ---
            # Create dispatch tensor (one-hot mask)
            # Indices are for dim S, num_classes = seq_len
            dispatch_tensor = F.one_hot(
                top_k_indices, num_classes=seq_len
            ).to(hidden_states.dtype)  # Shape: (B, E, C, S)

            # Gather expert inputs using einsum (Alg 1, step 3, x_in)
            # x_in = einsum('becs, bsd->becd', dispatch, x_p)
            expert_inputs = torch.einsum(
                "becs,bsd->becd", dispatch_tensor, hidden_states
            )  # Shape: (B, E, C, D)

            # Process tokens by each expert (Alg 1, step 3, x_e)
            expert_outputs_list = []
            for i in range(self.num_experts):
                # Input for expert i: (B, C, D)
                expert_i_input = expert_inputs[:, i, :, :]
                expert_outputs_list.append(self.experts[i](expert_i_input))
            expert_outputs_stacked = torch.stack(
                expert_outputs_list, dim=1
            )  # Shape: (B, E, C, D)

            # Combine expert outputs using einsum (Alg 1, step 3, x_out)
            # x_out = einsum('becs, bec, becd->bsd', dispatch, gating, x_e)
            # gating_scores = top_k_scores
            y = torch.einsum(
                "becs,bec,becd->bsd",
                dispatch_tensor,
                top_k_scores,
                expert_outputs_stacked,
            )  # Shape: (B, S, D)
        else:  # Inference Path: Sparse Gather/Scatter (more efficient)
            y = self.ec_infer(hidden_states, top_k_indices, top_k_scores)
        output = y.to(identity.dtype) + self.linear_layer(identity)

        return output  # Return only the final output

    def forward_by_type_topk(self, hidden_states):
        if self.without_experts:
            self.without_experts = False
            return self.linear_layer(hidden_states)

        assert LoRAMoE.type_embedding is not None, "LoRAMoE.type_embedding has not been set."
        type_embedding = LoRAMoE.type_embedding

        # --- 步骤1：检查与统一化 (Normalize to 3D) ---
        is_2d = hidden_states.ndim == 2
        if is_2d:
            # 将 [B, D] -> [B, 1, D]
            # 这样我们可以用同一套3D代码处理
            hidden_states = hidden_states.unsqueeze(1)

        # 经过LoRA专家层的原始输出
        out = self.linear_layer(hidden_states)
        bsz, seq_len, out_dim = out.shape
        in_dim = hidden_states.shape[-1]
        
        # --- 步骤2：统一的3D处理逻辑 (现在对2D和3D都适用) ---
        # pooled_hidden_state = torch.mean(hidden_states, dim=1)
        # gate_input = torch.cat([pooled_hidden_state, type_embedding], dim=-1)
        # 2.1 获取专家路由决策
        topk_idx, topk_weight, aux_loss = self.gate(type_embedding)
        if LoRAMoE.debug:
            self.latest_token_indices = topk_idx.detach()
            self.latest_token_weights = topk_weight.detach()
        # 检查批次大小
        if topk_idx.shape[0] != bsz:
            raise ValueError(f"Batch size mismatch: hidden_states ({bsz}) vs type_embedding ({topk_idx.shape[0]})")
        
        # 2.2 批量计算所有专家的输出
        # [B, S, D_in] -> [B*S, D_in]
        hidden_states_flat = hidden_states.view(-1, in_dim)
        
        # [NumExperts, B*S, D_out]
        all_expert_outputs = torch.stack(
            [expert(hidden_states_flat) for expert in self.experts], dim=0
        )

        # 2.3 高效地选择和聚合
        # [NumExperts, B*S, D_out] -> [B, S, NumExperts, D_out]
        expert_outputs = all_expert_outputs.view(len(self.experts), bsz, seq_len, out_dim).permute(1, 2, 0, 3)

        # [B, K] -> [B, 1, K, 1] -> [B, S, K, D_out]
        expanded_topk_idx = topk_idx.unsqueeze(1).unsqueeze(3).expand(-1, seq_len, -1, out_dim)
        
        # gathered_outputs 形状: [B, S, K, D_out]
        gathered_outputs = torch.gather(expert_outputs, 2, expanded_topk_idx)

        # 加权并求和
        weighted_outputs = gathered_outputs * topk_weight.unsqueeze(1).unsqueeze(3)
        final_expert_out = weighted_outputs.sum(dim=2) # 形状: [B, S, D_out]
        
        if self.training and aux_loss is not None:
            if self.orthogonal_reg_alpha:
                l_ortho = compute_orthogonality_loss(all_expert_outputs)
                aux_loss = (aux_loss + l_ortho * self.orthogonal_reg_alpha) * LoRAMoE.aux_loss_weight
            final_expert_out = AddAuxiliaryLoss.apply(final_expert_out, aux_loss)
        out += final_expert_out
        # --- 步骤3：检查与还原 (Denormalize) ---
        if is_2d:
            # 如果原始输入是2D的，把我们临时添加的seq_len维度去掉
            # [B, 1, D] -> [B, D]
            out = out.squeeze(1)
                
        return out


    def forward_by_type(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes the MoE output using type-based soft routing via einsum.
        This version is robustly designed to handle both 2D [B, D] and 3D [B, S, D] inputs.
        """
        if self.without_experts:
            self.without_experts = False
            return self.linear_layer(hidden_states)

        assert LoRAMoE.type_embedding is not None, "LoRAMoE.type_embedding has not been set."
        type_embedding = LoRAMoE.type_embedding # 形状: [B, D_in]

        # --- 步骤 0: 归一化输入形状 (Normalize Input Shape) ---
        is_2d = hidden_states.ndim == 2
        if is_2d:
            # 临时将 [B, D] -> [B, 1, D]
            # 这样后续所有代码都可以按 3D 逻辑统一处理
            hidden_states = hidden_states.unsqueeze(1)

        # --- 1. 获取路由权重 ---
        # gate 只返回一个张量: scores
        # 形状: [B, E] (bsz, num_experts)
        routing_weights = self.gate(type_embedding)
        self.latest_routing_weights = routing_weights

        # --- 2. 批量计算所有专家的输出 ---
        bsz, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        
        # all_expert_outputs 形状: [E, B*S, D_out]
        all_expert_outputs = torch.stack(
            [expert(hidden_states_flat) for expert in self.experts], dim=0
        )

        # --- 3. 使用 einsum 高效地进行加权求和 ---
        # routing_weights [B, E] 需要扩展到每个 token 上
        expanded_weights = routing_weights.repeat_interleave(seq_len, dim=0)
        
        # 转置专家输出，方便 einsum
        expert_outputs_permuted = all_expert_outputs.permute(1, 0, 2)
        # 我们使用更明确的字母: 'n' for tokens, 'e' for experts, 'd' for dimension
        # 规则: 'ne,ned->nd'
        #   - 输入1 ('ne'): [num_tokens, num_experts]
        #   - 输入2 ('ned'): [num_tokens, num_experts, out_dim]
        #   - 输出 ('nd'): [num_tokens, out_dim]
        #   - 'e' 维度被点积并求和
        final_expert_out_flat = torch.einsum(
            'ne,ned->nd', 
            expanded_weights, 
            expert_outputs_permuted
        )
        
        # --- 4. 组合最终结果 ---
        out_dim = self.linear_layer.out_features
        final_expert_out = final_expert_out_flat.view(bsz, seq_len, out_dim)
        # --- 步骤 5: 反归一化输出形状 (Denormalize Output Shape) ---
        if is_2d:
            # 如果原始输入是 2D 的，将输出从 [B, 1, D] 还原回 [B, D]
            final_expert_out = final_expert_out.squeeze(1)
        out = self.linear_layer(hidden_states) + final_expert_out
        return out


    def forward_by_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        An optimized, fully vectorized implementation of token-based MoE routing.
        It avoids Python loops by using torch.gather and supports hybrid input signals.
        """
        if self.without_experts:
            self.without_experts = False
            return self.linear_layer(hidden_states)

        # --- 1. 准备输入和获取维度 ---
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

        # --- 2. 准备混合输入并进行路由 ---
        gate_input = hidden_states_flat
        # Gate 将返回硬路由的结果: (indices, weights, aux_loss)
        topk_idx, topk_weight, aux_loss = self.gate(gate_input) # 形状均为 [N, k]

        if LoRAMoE.debug:
            self.latest_token_indices = topk_idx.detach()
            self.latest_token_weights = topk_weight.detach()

        # --- 3. 批量计算所有专家的输出 ---
        # all_expert_outputs 形状: [E, N, D_out] (num_experts, num_tokens, out_dim)
        all_expert_outputs = torch.stack(
            [expert(hidden_states_flat) for expert in self.experts], dim=0
        )
        out_dim = all_expert_outputs.shape[-1]

        # --- 4. 高效地选择和聚合专家输出 (Gather-Sum) ---
        # 将专家输出转置为 [N, E, D_out] 以便 gather
        all_expert_outputs_permuted = all_expert_outputs.permute(1, 0, 2)

        # 准备 gather 所需的索引
        # topk_idx [N, k] -> [N, k, 1] -> [N, k, D_out]
        indices_for_gather = topk_idx.unsqueeze(-1).expand(-1, -1, out_dim)

        # 并行地为每个 token 挑选出 k 个专家的输出
        # selected_outputs 形状: [N, k, D_out]
        selected_outputs = torch.gather(all_expert_outputs_permuted, 1, indices_for_gather)

        # 准备加权所需的权重
        # topk_weight [N, k] -> [N, k, 1]
        weights_for_sum = topk_weight.unsqueeze(-1)
        
        # 加权并沿 k 维度求和
        # expert_output_flat 形状: [N, D_out]
        expert_output_flat = (selected_outputs * weights_for_sum).sum(dim=1)

        # --- 5. 组合最终结果 ---
        # 恢复原始形状
        final_expert_out = expert_output_flat.view(*orig_shape[:-1], out_dim)        

        # --- 6. 应用辅助损失 ---
        if self.training and aux_loss is not None:
            if self.orthogonal_reg_alpha:
                l_ortho = compute_orthogonality_loss(all_expert_outputs)
                aux_loss = (aux_loss + l_ortho * self.orthogonal_reg_alpha) * LoRAMoE.aux_loss_weight
            final_expert_out = AddAuxiliaryLoss.apply(final_expert_out, aux_loss)
        final_out = self.linear_layer(hidden_states) + final_expert_out
        return final_out
    
    def forward_token_expert_race(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.without_experts:
            self.without_experts = False
            return self.linear_layer(hidden_states)

        orig_shape = hidden_states.shape
        in_features = self.linear_layer.in_features
        hidden_states_flat = hidden_states.reshape(-1, in_features)
        
        final_weights, lsim = self.gate(hidden_states_flat)

        all_expert_outputs = torch.stack(
            [expert(hidden_states_flat) for expert in self.experts], dim=0
        )
        
        expert_output_flat = torch.einsum(
            'ne,ned->nd',
            final_weights,
            all_expert_outputs.permute(1, 0, 2)
        )
        
        out_dim = self.linear_layer.out_features
        output_shape = orig_shape[:-1] + (out_dim,)
        final_expert_out = expert_output_flat.view(output_shape)

        if self.training and lsim is not None and lsim.item() > 0:
            if self.orthogonal_reg_alpha:
                l_ortho = compute_orthogonality_loss(all_expert_outputs)
                lsim = lsim + l_ortho * self.orthogonal_reg_alpha
            final_expert_out = AddAuxiliaryLoss.apply(final_expert_out, lsim)
        final_out = self.linear_layer(hidden_states) + final_expert_out
        return final_out
        

def get_mole_class(mole_config):
    if "mole_expert" not in mole_config:
        ExpertClass = LoRAMoE
    elif mole_config["mole_expert"] == "LoRAMoE":
        ExpertClass = LoRAMoE
    else:
        raise NotImplementedError
    return ExpertClass


def load_experts_weights(transformer, load_path, strict=True):
    experts_state_dict = torch.load(load_path)
    load_result = transformer.load_state_dict(experts_state_dict, strict=False)
    if strict:
        # missing = load_result.missing_keys
        unexpected = load_result.unexpected_keys
        # assert not missing, f"Missing keys: {missing}"
        assert not unexpected, f"Unexpected keys: {unexpected}"
    print("Loaded experts weights conditionom {}".format(load_path))


def random_set_expert_gate_status(transformer, mole_config, prob=0.1):
    ExpertClass = get_mole_class(mole_config)
    for _, module in transformer.named_modules():
        if isinstance(module, ExpertClass):
            module.gate.requires_grad = random.random() < prob


# mole_config = {
#     "mole_expert": "SparseMoleFeedForwardExpertRace",
#     "num_experts": 8,
#     "num_experts_per_tok": 2,
#     "rank": 4,
#     "alpha": 4,
#     "dropout": 0.0,
#     "train_mole_onlytrain_mole_only": True,
#     "svd_init": True}


def convert_to_lora_moe(transformer, mole_config, target_modules):
    """
    只根据给定的 index 列表决定哪些层 Token-Based，其余全部 Type-Based。
    - token_route_overrides_double: [int]，双流区插针层索引（如 0~18）
    - token_route_overrides_single: [int]，单流区插针层索引（如 0~37）
    """
    ExpertClass = get_mole_class(mole_config)
    MAX_DOUBLE_STREAM_LAYERS = 19
    MAX_SINGLE_STREAM_LAYERS = 38
    double_config = mole_config.get("token_route_overrides_double")
    if double_config == "all":
        tok_over_d = set(range(MAX_DOUBLE_STREAM_LAYERS))
    elif isinstance(double_config, list):
        tok_over_d = set(double_config)
    else:
        tok_over_d = set()
    single_config = mole_config.get("token_route_overrides_single")
    if single_config == "all":
        tok_over_s = set(range(MAX_SINGLE_STREAM_LAYERS))
    elif isinstance(single_config, list):
        tok_over_s = set(single_config)
    else:
        tok_over_s = set()

    print(f"Token-Based (double): {sorted(tok_over_d)}")
    print(f"Token-Based (single): {sorted(tok_over_s)}")
    print("All other layers: Type-Based")

    for name, module in list(transformer.named_modules()):
        if isinstance(module, torch.nn.Linear) and any(
            target_substring in name for target_substring in target_modules
        ):
            is_token_based = False
            if "single_transformer_blocks" in name:
                idx = int(name.split('.')[1])
                if idx in tok_over_s:
                    is_token_based = True
            elif "transformer_blocks" in name:
                idx = int(name.split('.')[1])
                if idx in tok_over_d:
                    is_token_based = True
           
            parent_module = transformer
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            attr_name = name_parts[-1]

            moe_mod = ExpertClass(
                module,
                num_experts=mole_config["num_experts"],
                num_experts_per_tok=mole_config["num_experts_per_tok"],
                rank=mole_config["rank"],
                alpha=mole_config["alpha"],
                train_mole_only=mole_config.get("train_mole_only", True),
                route_by_type=not is_token_based,   
                type_aux_loss_alpha=mole_config.get("type_aux_loss_alpha", 0.1), 
                token_aux_loss_alpha=mole_config.get("token_aux_loss_alpha", 0.01),
                orthogonal_reg_alpha=mole_config.get("orthogonal_reg_alpha", 0.1),
            )
            setattr(parent_module, attr_name, moe_mod)
            print("========>", name, moe_mod.route_by_type)


def save_mole(transformer, path, mole_config):
    from diffusers.pipelines import FluxPipeline
    from peft import get_peft_model_state_dict

    # path = path + "/" + adpter_name
    FluxPipeline.save_lora_weights(
        save_directory=path,
        transformer_lora_layers=get_peft_model_state_dict(
            model=transformer,
        ),
        safe_serialization=True,
    )
    save_experts_weights(path + "/mole_experts.pt", transformer, mole_config)
    
def set_expert_gate_status(transformer, mole_config, requires_grad):
    ExpertClass = get_mole_class(mole_config)
    for _, module in transformer.named_modules():
        if isinstance(module, ExpertClass):
            module.gate.requires_grad = requires_grad


def save_experts_weights(save_path, transformer, mole_config):
    trainable_params = {}
    ExpertClass = get_mole_class(mole_config)
    set_expert_gate_status(transformer, mole_config, requires_grad=True)
    for prefix, module in transformer.named_modules():
        if isinstance(module, ExpertClass):
            module_params = {
                prefix + "." + name: param
                for name, param in module.named_parameters()
                if param.requires_grad
            }
            trainable_params.update(module_params)
            for name, buf in module.named_buffers():
                trainable_params[prefix + "." + name] = buf
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trainable_params, save_path)
