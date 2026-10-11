To realize the neuro-symbolic, compiler-grounded system we've designed, the architecture cannot be a standard off-the-shelf Decoder-only LLM. It must be a **Multi-Modal, Actor-Critic Transformer with a Grammar-Constrained Policy Head**. 

Below is the detailed architectural blueprint for this model, which we'll call **NeuroCode-ASM**.

---

### 1. High-Level Architecture Overview
The model consists of four primary subsystems:
1.  **Shared Backbone:** A Sparse Mixture-of-Experts (MoE) Decoder-only Transformer.
2.  **Multi-Modal Input Aligner:** Projects both Source Code and Assembly into the same continuous latent space.
3.  **Grammar-Constrained Policy Head:** Outputs probabilities over Context-Free Grammar (CFG) production rules.
4.  **Value & Execution Grounding Head:** Estimates the quality of a partial AST and aligns it with the target Assembly semantics.

---

### 2. Component-Level Blueprint

#### A. Input & Tokenization Space
*   **Unified Vocabulary:** A single BPE tokenizer trained on both Source Code and Assembly/LLVM IR, plus special tokens: `<|asm_start|>`, `<|asm_end|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`.
*   **Modality Embeddings:** Besides standard token and RoPE positional embeddings, an embedding layer is added to distinguish whether the current token is Source Code or Assembly.
*   **YaRN RoPE:** Rotational Positional Embeddings scaled dynamically to handle repository-level contexts (up to 32k+ tokens).

#### B. The Shared Backbone (Sparse MoE Transformer)
Instead of a dense feed-forward network (FFN), the backbone uses a Sparse Mixture-of-Experts to handle the vast distributional shift between high-level languages (Python, Rust) and low-level Assembly.

*   **Attention:** Grouped-Query Attention (GQA) with FlashAttention-2.
*   **MoE FFN Layer:** 
    *   1 Shared Expert (always active, handles universal syntax like brackets).
    *   8 Routed Experts (activated via Top-2 gating). The router naturally learns to route Assembly tokens to hardware-specific experts and high-level code to logic-specific experts.
*   **Output:** A sequence of continuous hidden states $H \in \mathbb{R}^{L \times d}$ (where $L$ is sequence length, $d$ is hidden dimension).

#### C. Multi-Modal Assembly Grounding (Cross-Attention)
To ground the LLM in execution semantics, the hidden states of the Source Code must attend to the hidden states of the Target Assembly.

*   **Assembly Encoder Path:** The target Assembly is passed through the shared backbone to produce `asm_hidden_states`.
*   **Cross-Attention Module:** Source code hidden states act as *Queries*, and Assembly hidden states act as *Keys* and *Values*.
*   **Result:** The source code representations are mathematically infused with hardware-level execution flow.

#### D. The Neuro-Symbolic Heads (For MCTS)
This is where the architecture diverges from standard LLMs. Instead of a linear layer outputting token probabilities, it outputs **Grammar Rule probabilities** and **State Values**.

1.  **Grammar Rule Lookup Table:** An embedding layer $E_{rule} \in \mathbb{R}^{R \times d}$, where $R$ is the total number of production rules in the union of all language CFGs (e.g., Python's AST grammar + C's AST grammar).
2.  **Policy Head ($\pi_\theta$):**
    *   Takes the final hidden state $h_t$ at the current AST expansion node.
    *   Computes dot products between $h_t$ and all grammar rule embeddings: $\text{logits} = h_t \cdot E_{rule}^T$.
    *   **Grammar Masking:** A dynamic boolean mask is applied to $\text{logits}$, setting the probability of invalid grammar rules (for the current AST node) to $-\infty$.
    *   Softmax outputs the policy distribution over valid rules.
3.  **Value Head ($V_\theta$):**
    *   A multi-layer perceptron (MLP) that takes $h_t$ and outputs a scalar value estimating the expected Execution Reward (how closely the final code will match the target Assembly).

---

### 3. PyTorch Architecture Skeleton

Here is the PyTorch implementation of this specific architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    """Mixture of Experts FFN layer"""
    def __init__(self, d_model, num_experts=8, d_ff=4096, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        router_logits = self.router(x) # [B, L, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select Top-K experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (simplified, assumes efficient batched routing)
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i] # [B, L]
            weight = topk_weights[..., i].unsqueeze(-1) # [B, L, 1]
            # Gather and compute (in reality, use advanced indexing for efficiency)
            for b in range(x.size(0)):
                for l in range(x.size(1)):
                    e = expert_idx[b, l]
                    output[b, l] += weight[b, l] * self.experts[e](x[b, l])
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = SparseMoE(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None, kv_cross=None):
        # Self-Attention
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        
        # Cross-Attention (if Assembly grounding is provided)
        if kv_cross is not None:
            cross_out, _ = self.attn(x, kv_cross, kv_cross)
            x = self.norm1(x + cross_out)
            
        # Sparse FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class NeuroSymbolicCodeLLM(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layers=12, n_heads=16, num_grammar_rules=500):
        super().__init__()
        # 1. Inputs
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.modality_emb = nn.Embedding(2, d_model) # 0=Source, 1=Assembly
        
        # 2. Backbone
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        
        # 3. Neuro-Symbolic Heads
        self.rule_embeddings = nn.Embedding(num_grammar_rules, d_model)
        self.policy_head = nn.Linear(d_model, num_grammar_rules)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def encode(self, token_ids, modality_ids):
        """Shared backbone encoding"""
        x = self.token_emb(token_ids) + self.modality_emb(modality_ids)
        
        # In a real implementation, add RoPE here
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, src_tokens, src_modalities, asm_tokens=None, asm_modalities=None, current_ast_node_idx=None):
        # 1. Encode Target Assembly (if grounding)
        asm_hidden = None
        if asm_tokens is not None:
            asm_hidden = self.encode(asm_tokens, asm_modalities)
            
        # 2. Encode Source Code with Assembly Cross-Attention
        src_hidden = self.token_emb(src_tokens) + self.modality_emb(src_modalities)
        for block in self.blocks:
            # Pass asm_hidden as kv_cross for the cross-attention mechanism
            src_hidden = block(src_hidden, kv_cross=asm_hidden)
            
        # 3. MCTS Heads (operate on the hidden state of the node to expand)
        # current_ast_node_idx represents the position in the sequence we are expanding
        h_t = src_hidden[:, current_ast_node_idx, :] # [B, d_model]
        
        # Policy: log probabilities over grammar rules
        policy_logits = self.policy_head(h_t)
        
        # Value: estimate of execution reward
        value = self.value_head(h_t)
        
        return policy_logits, value

# --- Example Usage ---
# vocab_size = 32000 (Source + Assembly BPE)
# num_grammar_rules = 450 (Union of Python/C AST rules)
# model = NeuroSymbolicCodeLLM(vocab_size=32000, num_grammar_rules=450)
```

### 4. The Inference / MCTS Generation Loop
Because this architecture has a Policy Head and Value Head, inference is completely different from standard LLM `.generate()`.

1.  **Initialize:** The prompt is parsed into a partial AST. The unexpanded node is identified.
2.  **Forward Pass:** The partial code is passed through the model. The Policy Head outputs probabilities over valid grammar rules. The Value Head estimates the quality of the current state.
3.  **MCTS Expansion:** The search algorithm expands the AST using the policy distribution, selecting the most promising rules (UCT formula).
4.  **Rollout:** A quick random expansion finishes the AST. It is converted to source code and compiled.
5.  **Reward:** The generated Assembly is compared to the target Assembly. The reward is backpropagated through the MCTS tree, updating the Value Head's weights.
6.  **Output:** Once MCTS finds the highest-reward path, the final AST is unparsed into Source Code and output to the user.