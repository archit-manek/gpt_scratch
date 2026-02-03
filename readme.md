## Contents

- [Contents](#contents)
- [Data Loading](#data-loading)
- [GPT Data Batching \& Shapes](#gpt-data-batching--shapes)
  - [1. Tensors vs. Python Lists](#1-tensors-vs-python-lists)
  - [2. Key Dimensions (B, T, C)](#2-key-dimensions-b-t-c)
  - [3. Random Sampling Logic](#3-random-sampling-logic)
  - [4. Input (x) vs. Target (y)](#4-input-x-vs-target-y)
- [Autoregressive Loop](#autoregressive-loop)
  - [1. The Loop Setup](#1-the-loop-setup)
  - [2. The Forward Pass](#2-the-forward-pass)
  - [3. The "Focus" Step](#3-the-focus-step)
  - [4. Normalization (Softmax)](#4-normalization-softmax)
  - [5. Sampling (Multinomial)](#5-sampling-multinomial)
  - [6. The Update (Concatenation)](#6-the-update-concatenation)
  - [Visual Summary](#visual-summary)
- [Self Attention](#self-attention)
- [Hyperparameters Reference](#hyperparameters-reference)
- [1. Head (Single Self-Attention Head)](#1-head-single-self-attention-head)
  - [Purpose](#purpose)
  - [Code](#code)
  - [Line-by-Line Breakdown](#line-by-line-breakdown)
  - [Key Intuitions](#key-intuitions)
- [2. MultiHeadAttention](#2-multiheadattention)
  - [Purpose](#purpose-1)
  - [Code](#code-1)
  - [Line-by-Line Breakdown](#line-by-line-breakdown-1)
  - [Key Intuitions](#key-intuitions-1)
- [3. FeedForward](#3-feedforward)
  - [Purpose](#purpose-2)
  - [Code](#code-2)
  - [Line-by-Line Breakdown](#line-by-line-breakdown-2)
  - [Key Intuitions](#key-intuitions-2)
- [4. Block (Transformer Block)](#4-block-transformer-block)
  - [Purpose](#purpose-3)
  - [Code](#code-3)
  - [Line-by-Line Breakdown](#line-by-line-breakdown-3)
  - [Key Intuitions](#key-intuitions-3)
- [5. BigramLanguageModel (Main Model)](#5-bigramlanguagemodel-main-model)
  - [Purpose](#purpose-4)
  - [Code](#code-4)
  - [Line-by-Line Breakdown](#line-by-line-breakdown-4)
  - [Key Intuitions](#key-intuitions-4)
- [Architecture Diagram](#architecture-diagram)
- [Parameter Count Breakdown](#parameter-count-breakdown)
- [Key Concepts Summary](#key-concepts-summary)
- [Questions to Test Understanding](#questions-to-test-understanding)

---

[Google Colab (Karpathy)](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)

## Data Loading

---

- Reading the `input.txt` (Tiny Shakespeare).
- Building the vocabulary (sorted list of unique characters).
- Creating the `stoi` (string-to-integer) and `itos` mappings.

## GPT Data Batching & Shapes

---

### 1. Tensors vs. Python Lists

- **The Issue:** PyTorch tensors (e.g., `tensor(5)`) are not the same as Python integers (`5`).
- **The Fix:** Use `.tolist()` to extract values before passing them to standard Python functions (like string decoders).

```python
# Don't loop over the tensor directly
decode(out.tolist())
```

---

### 2. Key Dimensions (B, T, C)

In Mechanical Interpretability, always track the shapes:

- **B (Batch):** Number of independent sequences processed in parallel.
- **T (Time/Block Size):** The maximum context length (history) the model can see.
- **C (Channel/Embed):** The vector size representing a single token.

---

### 3. Random Sampling Logic

```python
ix = torch.randint(len(data) - block_size, (batch_size,))
```

- **Purpose:** Picks `batch_size` random starting points in the text.
- **Why Random?** It breaks the correlation of the data.
    - **Row 1** might be from Chapter 1.
    - **Row 2** might be from Chapter 10.
    - This forces the model to learn **general language rules** (grammar/syntax) rather than memorizing the plot sequence.

---

### 4. Input (x) vs. Target (y)

We create `x` and `y` by stacking rows based on the random indices (`ix`).

```python
# The Input (Context)
x = torch.stack([data[i : i+block_size] for i in ix])

# The Target (Next Token) - Shifted right by 1
y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
```

- **The Slicing:**
    - `x` sees characters at indices `[0, 1, 2, 3]`
    - `y` sees characters at indices `[1, 2, 3, 4]`
- **The Lesson:** For every row, the model learns `block_size` separate predictions simultaneously (e.g., given char 1 predicts 2, given chars 1-2 predicts 3, etc.).

---

## Autoregressive Loop

This is the **Autoregressive Loop**. It is the heartbeat of GPT. "Autoregressive" just means "using my own past outputs as my future inputs."

Here is the breakdown, tracking the data shapes for a single batch (`B=1`) and a vocab size of `65`.

### 1. The Loop Setup

Python

```python
for _ in range(max_new_tokens):
```

- **Goal:** We want to generate `max_new_tokens` (e.g., 100) characters.
- **Method:** We will do this one character at a time. Predicting 100 characters at once is impossible; predicting 1 character is doable.

### 2. The Forward Pass

Python

```python
logits, loss = self(idx)
```

- **What it does:** Runs the current sequence `idx` through the model.
- **Input (`idx`):** Shape `(1, T)`. e.g., `[15, 42, 10...]`.
- **Output (`logits`):** Shape `(1, T, 65)`.
    - **Inefficiency Alert:** The model re-calculates the prediction for *every single character* in the sequence, even though we only care about the last one. (Later, "KV Caching" fixes this, but for now, we accept the re-calculation).

### 3. The "Focus" Step

Python

```python
logits = logits[:, -1, :]
```

- **The Logic:** The model output gives us a prediction for what comes after char 1, what comes after char 2, etc. We only care about **what comes after the very last character**.
- **The Shape Change:**
    - From: `(1, T, 65)` $\rightarrow$ 3D Tensor
    - To: `(1, 65)` $\rightarrow$ 2D Tensor
- We now have a row of 65 scores representing the likelihood of the **next** character.

### 4. Normalization (Softmax)

Python

```python
probs = F.softmax(logits, dim=1)
```

- **Math:** Exponentiate every score and divide by the sum ($e^x / \sum e^x$).
- **Result:** `logits` (which can be negative or huge numbers like -5.0 or 12.0) turn into `probs` (which are between 0.0 and 1.0 and sum to exactly 1).
- **Interpretation:** This is now a probability distribution. e.g., "There is a 10% chance the next letter is 'a', 80% chance it is 'e', etc."

### 5. Sampling (Multinomial)

Python

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

- **Why not just pick the highest score?** If you always pick the highest score (`argmax`), the model becomes deterministic and repetitive (it might get stuck in loops like "I went to the the the...").
- **What this does:** It rolls a weighted die.
    - If 'e' has 0.8 probability, it will pick 'e' 80% of the time.
    - But 20% of the time, it might pick 'a' or something else.
- **Output (`idx_next`):** Shape `(1, 1)`. A single integer wrapped in a tensor.

### 6. The Update (Concatenation)

Python

```python
idx = torch.cat((idx, idx_next), dim=1)
```

- **The "Auto" in Autoregressive:** We take the new character we just found (`idx_next`) and glue it onto the end of our history (`idx`).
- **Shape:** `(1, T)` becomes `(1, T+1)`.
- **Next Loop:** When the loop restarts, `self(idx)` will now see this new character as part of the context and use it to predict the *next* one.

### Visual Summary

1. **Read** history.
2. **Predict** next char scores.
3. **Sample** one char.
4. **Append** char to history.
5. **Repeat.**

---

## Self Attention

```python
# Version 3: Softmax
tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T, T))
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=-1)
xbow3 = weights @ x
```

- W initialize the "interaction strength" between all tokens to **0**
    - This is a placeholder. In real Self-Attention, this won't be zeros; it will be the result of a "compatibility calculation" (Queries dot Keys).
    - For now, "0" means "I like everyone equally.”
- Masking
    - A token at time step 5 can only know about steps 1, 2, 3, and 4. It **cannot** know about step 6.
        - `tril` creates a triangle of 1s (allowed) and 0s (forbidden).
        - `masked_fill` looks at all the forbidden spots (where `tril == 0`) and sets the weight to **negative infinity (`inf`)**.
- The Softmax (The "Normalization")
    - Softmax takes numbers and turns them into probabilities that sum to 1.
        - $e^0 = 1$
        - $e^{-\infty} = 0$
    - By setting the future to `-inf`, we force the probability of attending to the future to be **exactly zero**.

## Hyperparameters Reference

```python
batch_size = 16      # how many independent sequences to process in parallel
block_size = 32      # maximum context length for predictions
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64          # embedding dimension
n_head = 4           # number of attention heads
n_layer = 4          # number of transformer blocks
dropout = 0.0

```

---

## 1. Head (Single Self-Attention Head)

### Purpose

One head of self-attention. This is the fundamental building block that allows tokens to "communicate" with each other. Each head learns to look for different patterns/relationships between tokens.

### Code

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,T)

        # mask out the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax to get probabilities
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,head_size)
        return out

```

### Line-by-Line Breakdown

**`__init__` method:**

| Line | What it does |
| --- | --- |
| `self.key = nn.Linear(n_embd, head_size, bias=False)` | Projects input to "what do I contain?" representation |
| `self.query = nn.Linear(n_embd, head_size, bias=False)` | Projects input to "what am I looking for?" representation |
| `self.value = nn.Linear(n_embd, head_size, bias=False)` | Projects input to "what do I communicate?" representation |
| `self.register_buffer('tril', ...)` | Creates lower triangular mask (not a parameter, but moves with model to GPU) |
| `self.dropout = nn.Dropout(dropout)` | Regularization - randomly zeros some attention weights during training |

**`forward` method:**

| Line | Shape | What it does |
| --- | --- | --- |
| `B, T, C = x.shape` | — | Unpack batch size, sequence length, channels |
| `k = self.key(x)` | (B,T,head_size) | Compute keys for all positions |
| `q = self.query(x)` | (B,T,head_size) | Compute queries for all positions |
| `wei = q @ k.transpose(-2,-1)` | (B,T,T) | Dot product: how much does each query attend to each key? |
| `* C**-0.5` | (B,T,T) | Scale by √d_k to prevent softmax saturation |
| `wei.masked_fill(...)` | (B,T,T) | Set future positions to -inf (causal masking) |
| `F.softmax(wei, dim=-1)` | (B,T,T) | Normalize to probabilities (rows sum to 1) |
| `v = self.value(x)` | (B,T,head_size) | Compute values for all positions |
| `out = wei @ v` | (B,T,head_size) | Weighted sum of values based on attention |

### Key Intuitions

- **Query-Key-Value analogy**: Think of it like a search engine. Query = "search terms", Key = "document titles", Value = "document content". You match queries to keys, then retrieve the corresponding values.
- **Why scale by √d_k?** Without scaling, dot products grow large with dimension, pushing softmax into regions with tiny gradients. Scaling keeps variance ~1.
- **Causal mask**: Token at position t can only attend to positions 0, 1, ..., t. This prevents "cheating" by looking at future tokens during training.

---

## 2. MultiHeadAttention

### Purpose

Runs multiple attention heads in parallel and concatenates their outputs. Different heads can learn to attend to different types of relationships (e.g., one head might learn syntax, another semantics).

### Code

```python
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

```

### Line-by-Line Breakdown

**`__init__` method:**

| Line | What it does |
| --- | --- |
| `self.heads = nn.ModuleList([...])` | Create `num_heads` independent attention heads |
| `self.proj = nn.Linear(n_embd, n_embd)` | Projection back into residual pathway |
| `self.dropout = nn.Dropout(dropout)` | Regularization after projection |

**`forward` method:**

| Line | Shape | What it does |
| --- | --- | --- |
| `[h(x) for h in self.heads]` | List of (B,T,head_size) | Run each head independently |
| `torch.cat(..., dim=-1)` | (B,T,n_embd) | Concatenate along channel dimension |
| `self.proj(out)` | (B,T,n_embd) | Linear projection (mixes information across heads) |
| `self.dropout(...)` | (B,T,n_embd) | Apply dropout for regularization |

### Key Intuitions

- **Why multiple heads?** Each head has its own Q, K, V projections, so can learn different attention patterns. Concatenating gives the model multiple "representation subspaces."
- **Math check**: `n_head=4`, `head_size = n_embd // n_head = 64 // 4 = 16`. Each head outputs 16 dims, concatenated = 64 dims = `n_embd`. ✓
- **Projection layer**: After concatenation, the projection layer allows heads to communicate and mix their findings.

---

## 3. FeedForward

### Purpose

A simple MLP applied independently to each position. After attention (tokens communicating), this lets each token "think" about what it gathered. Adds computational capacity.

### Code

```python
class FeedFoward(nn.Module):
    """ a simple feed-forward layer """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

```

### Line-by-Line Breakdown

| Line | Shape transformation | What it does |
| --- | --- | --- |
| `nn.Linear(n_embd, 4 * n_embd)` | (B,T,64) → (B,T,256) | Expand to higher dimension |
| `nn.ReLU()` | (B,T,256) → (B,T,256) | Non-linearity |
| `nn.Linear(4 * n_embd, n_embd)` | (B,T,256) → (B,T,64) | Project back down |
| `nn.Dropout(dropout)` | (B,T,64) → (B,T,64) | Regularization |

### Key Intuitions

- **Why 4x expansion?** This is a design choice from the original Transformer paper. The inner dimension of 4x gives the network more computational capacity. Think of it as giving each position a "bigger brain" to process information.
- **Position-wise**: The same MLP is applied to every position independently. No communication between positions here—that's what attention is for.
- **ReLU vs GELU**: Original Transformer used ReLU. GPT-2/3 use GELU. Both work; GELU is smoother.

---

## 4. Block (Transformer Block)

### Purpose

One complete transformer layer: self-attention (communication) followed by feed-forward (computation). Uses residual connections and layer normalization for stable training.

### Code

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

```

### Line-by-Line Breakdown

**`__init__` method:**

| Line | What it does |
| --- | --- |
| `head_size = n_embd // n_head` | Calculate size per head (64 // 4 = 16) |
| `self.sa = MultiHeadAttention(...)` | Multi-head self-attention sublayer |
| `self.ffwd = FeedFoward(n_embd)` | Feed-forward sublayer |
| `self.ln1 = nn.LayerNorm(n_embd)` | Layer norm before attention (Pre-LN) |
| `self.ln2 = nn.LayerNorm(n_embd)` | Layer norm before feed-forward |

**`forward` method:**

| Line | What it does |
| --- | --- |
| `x = x + self.sa(self.ln1(x))` | LayerNorm → Attention → Residual add |
| `x = x + self.ffwd(self.ln2(x))` | LayerNorm → FFN → Residual add |

### Key Intuitions

- **Pre-LN vs Post-LN**: Original Transformer did `x = LN(x + sublayer(x))` (Post-LN). This code does `x = x + sublayer(LN(x))` (Pre-LN). Pre-LN is more stable for training deep networks.
- **Residual connections (`x = x + ...`)**: Create "gradient highways." Gradients can flow directly backward through the `+`, preventing vanishing gradients in deep networks. Also allows layers to learn "deltas" on top of the identity.
- **Communication then computation**: Attention = "gather information from other tokens." FFN = "process what I gathered." This pattern repeats through all layers.

---

## 5. BigramLanguageModel (Main Model)

### Purpose

The complete GPT model. Combines token embeddings, positional embeddings, a stack of transformer blocks, and a final projection to vocabulary logits.

### Code

```python
class BigramLangaugeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

```

### Line-by-Line Breakdown

**`__init__` method:**

| Line | Shape | What it does |
| --- | --- | --- |
| `self.token_embedding_table` | (vocab_size, n_embd) | Lookup table: token index → vector |
| `self.position_embedding_table` | (block_size, n_embd) | Lookup table: position → vector |
| `self.blocks` | — | Stack of n_layer transformer blocks |
| `self.ln_f` | — | Final layer norm (Pre-LN architecture) |
| `self.lm_head` | (n_embd, vocab_size) | Project embeddings to vocab logits |

**`forward` method:**

| Line | Shape | What it does |
| --- | --- | --- |
| `B, T = idx.shape` | — | Get batch size and sequence length |
| `tok_emb = self.token_embedding_table(idx)` | (B,T,C) | Look up token embeddings |
| `pos_emb = self.position_embedding_table(torch.arange(T))` | (T,C) | Look up position embeddings for positions 0..T-1 |
| `x = tok_emb + pos_emb` | (B,T,C) | Combine (broadcasts over batch) |
| `x = self.blocks(x)` | (B,T,C) | Pass through all transformer blocks |
| `x = self.ln_f(x)` | (B,T,C) | Final layer normalization |
| `logits = self.lm_head(x)` | (B,T,vocab_size) | Project to vocabulary size |
| `logits.view(B*T, C)` | (B*T, C) | Flatten for cross_entropy |
| `targets.view(B*T)` | (B*T,) | Flatten targets to match |
| `F.cross_entropy(logits, targets)` | scalar | Compute loss |

**`generate` method:**

| Line | What it does |
| --- | --- |
| `idx_cond = idx[:, -block_size:]` | Crop context to max length model can handle |
| `logits, loss = self(idx_cond)` | Forward pass |
| `logits = logits[:, -1, :]` | Take logits for last position only |
| `probs = F.softmax(logits, dim=1)` | Convert to probabilities |
| `idx_next = torch.multinomial(probs, num_samples=1)` | Sample next token |
| `idx = torch.cat((idx, idx_next), dim=1)` | Append to sequence |

### Key Intuitions

- **Token + Position embeddings**: The model needs to know both *what* token is at each position AND *where* it is. Adding these together is simple but effective. (GPT-2/3 use learned positional embeddings like this; transformers can also use sinusoidal or rotary embeddings.)
- **Why final LayerNorm?** With Pre-LN architecture, there's no normalization after the last block's residual. The final LN ensures the representations going into lm_head are normalized.
- **Name "BigramLanguageModel"**: Karpathy starts with a simple bigram model and evolves it into GPT. The name is a bit of a misnomer for the final version—it's a full transformer!
- **Cross-entropy loss**: Each position predicts the next token. We compute cross-entropy loss across all positions and average.

---

## Architecture Diagram

```
Input tokens: [idx]
       ↓
Token Embedding: (B,T) → (B,T,n_embd)
       +
Position Embedding: (T,) → (T,n_embd) → broadcast → (B,T,n_embd)
       ↓
       x = tok_emb + pos_emb
       ↓
┌─────────────────────────────────┐
│         Block × n_layer          │
│  ┌───────────────────────────┐  │
│  │ LayerNorm                 │  │
│  │     ↓                     │  │
│  │ MultiHeadAttention        │  │
│  │     ↓                     │  │
│  │ + (residual)              │  │
│  │     ↓                     │  │
│  │ LayerNorm                 │  │
│  │     ↓                     │  │
│  │ FeedForward               │  │
│  │     ↓                     │  │
│  │ + (residual)              │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
       ↓
Final LayerNorm
       ↓
Linear (lm_head): (B,T,n_embd) → (B,T,vocab_size)
       ↓
Logits → Softmax → Probabilities → Sample/Argmax → Next Token

```

---

## Parameter Count Breakdown

With `n_embd=64, n_head=4, n_layer=4, vocab_size=65, block_size=32`:

| Component | Parameters |
| --- | --- |
| Token embeddings | 65 × 64 = 4,160 |
| Position embeddings | 32 × 64 = 2,048 |
| Per Head (K,Q,V) | 3 × (64 × 16) = 3,072 |
| Per MultiHead (4 heads + proj) | 4 × 3,072 + 64×64 = 16,384 |
| Per FFN | 64×256 + 256×64 = 32,768 |
| Per Block (MHA + FFN + 2×LN) | 16,384 + 32,768 + 2×128 = 49,408 |
| 4 Blocks | 4 × 49,408 = 197,632 |
| Final LN | 128 |
| lm_head | 64 × 65 = 4,160 |
| **Total** | **~0.21M** |

---

## Key Concepts Summary

| Concept | What it does | Why it matters |
| --- | --- | --- |
| **Self-Attention** | Tokens look at each other | Captures relationships regardless of distance |
| **Causal Masking** | Prevents looking at future | Enables autoregressive generation |
| **Multi-Head** | Multiple attention patterns | Richer representations |
| **Residual Connections** | `x = x + sublayer(x)` | Stable gradients, easier optimization |
| **Layer Normalization** | Normalize activations | Stable training |
| **Position Embeddings** | Encode token positions | Attention is permutation-invariant without this |
| **Feed-Forward** | Per-position MLP | Adds computational capacity |

---

## Questions to Test Understanding

1. Why do we scale attention scores by `1/√d_k`?
2. What would happen if we removed the causal mask during training?
3. Why use residual connections? What problem do they solve?
4. Why Pre-LN instead of Post-LN?
5. What's the purpose of the projection layer in MultiHeadAttention?
6. Why does the FFN expand to 4× the embedding dimension?
7. How does the model know token order if attention is permutation-invariant?