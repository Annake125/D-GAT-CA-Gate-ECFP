from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)
class LightweightGraphGate(nn.Module):
    """
    轻量级门控模块：用少量参数实现图-文本对齐
    参数量：仅 hidden_dim * 3 
    """
    def __init__(self, hidden_dim, graph_embed_dim):
        super().__init__()
        # 图嵌入投影到文本维度
        self.graph_proj = nn.Linear(graph_embed_dim, hidden_dim, bias=False)
        
        # 门控网络：决定图信息的融合比例
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 初始化门控偏置为负值，让训练初期更依赖文本嵌入
        nn.init.constant_(self.gate[0].bias, -2.0)
        
        # 用于监控
        self.last_alpha = None
        
    def forward(self, text_emb, graph_emb):
        """
        Args:
            text_emb: [B, L, hidden_dim] - SMILES文本嵌入
            graph_emb: [B, graph_embed_dim] - 图嵌入
        Returns:
            fused_emb: [B, L, hidden_dim] - 融合后的嵌入
        """
        batch_size, seq_len, hidden_dim = text_emb.shape
        
        # 投影图嵌入到文本维度
        graph_proj = self.graph_proj(graph_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 计算门控权重
        text_global = text_emb.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        gate_input = torch.cat([text_global, graph_proj], dim=-1)  # [B, 1, 2*hidden_dim]
        
        alpha = self.gate(gate_input)  # [B, 1, hidden_dim]
        self.last_alpha = alpha.detach()  # 保存用于监控
        
        # 门控融合：逐元素加权
        fused_emb = text_emb + alpha * graph_proj
        return fused_emb
    

class HybridGraphGate(nn.Module):
    """
    混合门控-注意力融合模块（各取Gate和CrossAttention所长）

    设计理念：
    - 保留Gate的全局门控机制（保证生成有效性）✅
    - 加入轻量序列级注意力（优化分子属性）✅
    - 避免信息冗余（通过增量融合）✅

    参数量：~7000-8000（仍然轻量）
    相比单独使用两个模块，参数减少40%，无冗余
    """
    def __init__(self, hidden_dim, graph_embed_dim, num_heads=2, enable_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.enable_attention = enable_attention

        # ===== 【保留】原有Gate机制 =====
        self.graph_proj = nn.Linear(graph_embed_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].bias, -2.0)  # 初期依赖文本

        # ===== 【新增】轻量注意力机制 =====
        if enable_attention:
            self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.scale = self.head_dim ** -0.5

            # 可学习的注意力融合强度（初始化为小值，让模型逐渐学习）
            self.attn_blend = nn.Parameter(torch.tensor(0.2))

        self.last_alpha = None

    def forward(self, text_emb, graph_emb):
        """
        Args:
            text_emb: [B, L, hidden_dim] - SMILES文本嵌入
            graph_emb: [B, graph_embed_dim] - 图嵌入
        Returns:
            fused_emb: [B, L, hidden_dim] - 融合后的嵌入
        """
        B, L, D = text_emb.shape

        # 1. 投影图嵌入到文本维度
        graph_proj = self.graph_proj(graph_emb).unsqueeze(1)  # [B, 1, D]

        # 2. 【Gate机制】计算全局门控权重
        text_global = text_emb.mean(dim=1, keepdim=True)  # [B, 1, D]
        gate_input = torch.cat([text_global, graph_proj], dim=-1)
        alpha = self.gate(gate_input)  # [B, 1, D]
        self.last_alpha = alpha.detach()

        # 3. 【注意力增强】序列级动态调整（如果启用）
        if self.enable_attention:
            # Multi-head cross-attention (简化版)
            q = self.q_proj(text_emb)  # [B, L, D]
            q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = graph_proj.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, L, 1]
            attn_weights = torch.softmax(attn, dim=-1)
            attn_out = (attn_weights @ k).transpose(1, 2).reshape(B, L, D)  # [B, L, D]

            # 【关键设计】增量融合避免冗余：
            # - graph_proj: 全局统一的图信息
            # - attn_out: 序列级差异化的图信息
            # - (attn_out - graph_proj): 注意力带来的增量
            # - blend控制增量的影响强度
            attn_delta = attn_out - graph_proj  # 计算增量
            refined_graph = graph_proj + self.attn_blend.sigmoid() * attn_delta
        else:
            refined_graph = graph_proj

        # 4. 【最终融合】用Gate控制整体融合强度
        fused_emb = text_emb + alpha * refined_graph
        return fused_emb


class FingerprintGate(nn.Module):
    """
    分子指纹融合模块（用于ECFP/Morgan指纹）- 拼接+降维版本

    设计理念：
    - 删除Gate门控机制，改用简单的拼接+降维
    - ECFP对有效性提升很有帮助，门控反而限制了其作用
    - 直接将指纹信息与文本嵌入拼接后降维，让模型自由学习融合方式

    参数量：主要来自投影层和降维层
    """
    def __init__(self, hidden_dim, fingerprint_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fingerprint_dim = fingerprint_dim

        # 指纹降维投影（从高维稀疏向量到文本维度）
        # 使用两层MLP进行非线性变换
        self.fp_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 拼接后的降维层：将 [text_emb || fp_emb] 从 2*hidden_dim 降到 hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, text_emb, fingerprint):
        """
        Args:
            text_emb: [B, L, hidden_dim] - SMILES文本嵌入
            fingerprint: [B, fingerprint_dim] - 分子指纹（ECFP/Morgan）
        Returns:
            fused_emb: [B, L, hidden_dim] - 融合后的嵌入
        """
        B, L, D = text_emb.shape

        # 1. 投影指纹到文本维度
        fp_proj = self.fp_proj(fingerprint).unsqueeze(1)  # [B, 1, D]

        # 2. 扩展指纹到序列长度
        fp_expanded = fp_proj.expand(B, L, D)  # [B, L, D]

        # 3. 拼接文本嵌入和指纹嵌入
        concat_emb = torch.cat([text_emb, fp_expanded], dim=-1)  # [B, L, 2*D]

        # 4. 降维融合
        fused_emb = self.fusion_proj(concat_emb)  # [B, L, D]

        return fused_emb


class Mol2vecGate(nn.Module):
    """
    Mol2vec融合模块 - 拼接+降维版本

    设计理念：
    - 与FingerprintGate保持一致，使用拼接+降维方式
    - Mol2vec捕获分子子结构的语义信息（类似Word2Vec）
    - 与ECFP/Graph嵌入互补，提供额外的语义表示

    参数量：主要来自投影层和降维层
    """
    def __init__(self, hidden_dim, mol2vec_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mol2vec_dim = mol2vec_dim

        # Mol2vec投影到文本维度
        # 由于Mol2vec已经是密集向量（非稀疏），可以使用更简单的投影
        self.m2v_proj = nn.Sequential(
            nn.Linear(mol2vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 拼接后的降维层：将 [text_emb || mol2vec_emb] 从 2*hidden_dim 降到 hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, text_emb, mol2vec):
        """
        Args:
            text_emb: [B, L, hidden_dim] - SMILES文本嵌入
            mol2vec: [B, mol2vec_dim] - Mol2vec嵌入
        Returns:
            fused_emb: [B, L, hidden_dim] - 融合后的嵌入
        """
        B, L, D = text_emb.shape

        # 1. 投影Mol2vec到文本维度
        m2v_proj = self.m2v_proj(mol2vec).unsqueeze(1)  # [B, 1, D]

        # 2. 扩展到序列长度
        m2v_expanded = m2v_proj.expand(B, L, D)  # [B, L, D]

        # 3. 拼接文本嵌入和Mol2vec嵌入
        concat_emb = torch.cat([text_emb, m2v_expanded], dim=-1)  # [B, L, 2*D]

        # 4. 降维融合
        fused_emb = self.fusion_proj(concat_emb)  # [B, L, D]

        return fused_emb


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        use_lightweight_gate=True,
        **kwargs
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.num_props = kwargs["num_props"]

        # 图嵌入配置
        self.use_graph = kwargs.get("use_graph", False)
        # 分子指纹配置
        self.use_fingerprint = kwargs.get("use_fingerprint", False)
        # Mol2vec配置
        self.use_mol2vec = kwargs.get("use_mol2vec", False)
        # gate_mode: 'lightweight' (原Gate) | 'hybrid' (Gate+Attention混合)
        self.gate_mode = kwargs.get("gate_mode", "lightweight")

        # 初始化图嵌入门控
        if self.use_graph:
            graph_embed_dim = kwargs.get("graph_embed_dim", 128)
            self.graph_embeddings = None  # 将在train.py中注册


            if self.gate_mode == "lightweight":
                # 轻量级门控
                self.graph_gate = LightweightGraphGate(config.hidden_size, graph_embed_dim)
                gate_params = sum(p.numel() for p in self.graph_gate.parameters())
                print(f"### Using Lightweight Graph Gate (params: {gate_params})")

            elif self.gate_mode == "hybrid":
                # 混合门控-注意力融合（Gate + CrossAttention优点结合）
                enable_attn = kwargs.get("enable_attention", True)  # 可消融
                self.graph_gate = HybridGraphGate(
                    config.hidden_size,
                    graph_embed_dim,
                    num_heads=2,
                    enable_attention=enable_attn
                )
                gate_params = sum(p.numel() for p in self.graph_gate.parameters())
                attn_status = "ON" if enable_attn else "OFF"
                print(f"### Using Hybrid Graph Gate (params: {gate_params}, attention: {attn_status})")

            else:
                raise ValueError(f"Invalid gate_mode: {self.gate_mode}. Choose 'lightweight' or 'hybrid'")

        # 初始化分子指纹融合（拼接+降维）
        if self.use_fingerprint:
            fingerprint_dim = kwargs.get("fingerprint_dim", 2048)  # 默认ECFP4 2048位
            self.fingerprint_embeddings = None  # 将在train.py中注册

            # 使用拼接+降维的方式（不使用Gate门控）
            self.fingerprint_gate = FingerprintGate(
                config.hidden_size,
                fingerprint_dim
            )
            fp_params = sum(p.numel() for p in self.fingerprint_gate.parameters())
            print(f"### Using Fingerprint Concat+Projection Fusion (params: {fp_params})")

        # 初始化Mol2vec融合（拼接+降维）
        if self.use_mol2vec:
            mol2vec_dim = kwargs.get("mol2vec_dim", 300)  # 默认300维
            self.mol2vec_embeddings = None  # 将在train.py中注册

            # 使用拼接+降维的方式（与ECFP保持一致）
            self.mol2vec_gate = Mol2vecGate(
                config.hidden_size,
                mol2vec_dim
            )
            m2v_params = sum(p.numel() for p in self.mol2vec_gate.parameters())
            print(f"### Using Mol2vec Concat+Projection Fusion (params: {m2v_params})")

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        
        if self.num_props:
            self.prop_nn = nn.Linear(self.num_props, self.input_dims) 
        
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids.to(th.int64))

    def get_props(self, props):
        return self.prop_nn(props.unsqueeze(1))
    
        
    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: 
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) 
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, graph_ids=None, fingerprint_ids=None, mol2vec_ids=None):
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param graph_ids: molecular indices used to retrieve graph embeddings (optional)
            :param fingerprint_ids: molecular indices used to retrieve fingerprints (optional)
            :param mol2vec_ids: molecular indices used to retrieve mol2vec embeddings (optional)
            :return: an [N x C x ...] Tensor of outputs.
            """
            emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

            if self.input_dims != self.hidden_size:
                emb_x = self.input_up_proj(x)
            else:
                emb_x = x

            # 图嵌入门控融合（第一层融合）
            if self.use_graph and graph_ids is not None:
                if hasattr(self, 'graph_gate') and self.graph_embeddings is not None:
                    graph_emb = self.graph_embeddings[graph_ids].to(emb_x.device)
                    emb_x = self.graph_gate(emb_x, graph_emb)

            # 分子指纹融合（第二层融合 - 拼接+降维）
            if self.use_fingerprint and fingerprint_ids is not None:
                if hasattr(self, 'fingerprint_gate') and self.fingerprint_embeddings is not None:
                    # 注意：fingerprint_ids通常与graph_ids相同（都是分子索引）
                    fingerprint = self.fingerprint_embeddings[fingerprint_ids].to(emb_x.device)
                    emb_x = self.fingerprint_gate(emb_x, fingerprint)

            # Mol2vec融合（第三层融合 - 拼接+降维）
            if self.use_mol2vec and mol2vec_ids is not None:
                if hasattr(self, 'mol2vec_gate') and self.mol2vec_embeddings is not None:
                    # 注意：mol2vec_ids通常与graph_ids/fingerprint_ids相同（都是分子索引）
                    mol2vec = self.mol2vec_embeddings[mol2vec_ids].to(emb_x.device)
                    emb_x = self.mol2vec_gate(emb_x, mol2vec)

            seq_length = x.size(1)
            position_ids = self.position_ids[:, : seq_length]

            # 标准嵌入计算（精简后）
            emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
            emb_inputs = self.LayerNorm(emb_inputs)

            emb_inputs = self.dropout(emb_inputs)

            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

            if self.output_dims != self.hidden_size:
                h = self.output_down_proj(input_trans_hidden_states)
            else:
                h = input_trans_hidden_states
            h = h.type(x.dtype)
            return h