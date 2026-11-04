# Attention on a napkin

The code is in [`tte/attention.py`](../tte/attention.py). The tests in
[`tests/test_attention.py`](../tests/test_attention.py) check it against
`torch.nn.functional.scaled_dot_product_attention`.

## The idea in one paragraph

Each position in a sequence asks a question, and every position it is allowed to look at
offers an answer. The question is a vector called the **query** (q). Each position also has a
**key** (k), which says what it can answer, and a **value** (v), which is what it hands over.
Position i compares its query with every key using a dot product: a large dot product means
"this is relevant to me". The scores go through a softmax so they become weights that are
positive and add up to 1. The output for position i is the weighted average of the values.
So attention is a soft, learned lookup: instead of picking one row of a table, take a blend
of all rows, weighted by how well each key matches the query.

## The formula

For a sequence of T positions, stack the queries, keys and values as rows of matrices
Q (T x d), K (T x d) and V (T x d_v):

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d) + M ) V

- `Q K^T` is a T x T table of scores: row i, column j says how much position i cares about j.
- `/ sqrt(d)`: a dot product of two random d-dimensional vectors has a spread of about
  sqrt(d). Without the division, large d pushes the softmax towards a one-hot choice whose
  gradient is almost zero, and training stalls. Dividing keeps the scores around unit scale.
- `M` is the mask. For a language model that predicts the next character, position i must not
  see positions after it, or it could copy the answer. So `M[i, j] = 0` when j <= i and
  minus infinity when j > i. After the softmax, minus infinity becomes weight 0. This is the
  **causal mask**: the table is lower triangular.
- `softmax` runs along each row, so each position's weights add up to 1.
- Multiplying by V mixes the values with those weights.

A worked example with T = 3 and the causal mask: position 1 can only attend to itself, so its
output is exactly v1. Position 2 blends v1 and v2. Position 3 blends v1, v2 and v3. Changing
the third character can never change what positions 1 and 2 compute;
`test_future_tokens_cannot_change_the_past` checks exactly that.

## Where Q, K and V come from

They are three learned linear maps of the same input x: `Q = x W_q`, `K = x W_k`,
`V = x W_v`. That is why it is called **self**-attention: the sequence attends to itself.
In the code one `nn.Linear(d_model, 3 * d_model)` computes all three at once.

## Multi-head

One set of weights can only express one notion of "relevant". Multi-head attention splits
the model width d_model into h heads of size d_model / h, runs attention in each head
independently (one head might track the previous letter, another the start of the word),
concatenates the h outputs and mixes them with one more linear map, `W_o`.
The cost is the same as one big head, because each head is smaller.

## Position

Attention by itself ignores order: shuffle the inputs and the outputs shuffle the same way.
So the model adds a position signal to each input embedding. This repository uses the
sinusoidal encoding from the original transformer paper:

    PE[pos, 2i]   = sin(pos / 10000^(2i / d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d))

Each pair of dimensions is a clock that ticks at its own speed, from fast to very slow, so
every position gets a distinct pattern and nearby positions get similar ones.

## A block, and the whole model

A decoder-only transformer is a stack of identical blocks. This repository uses the
**pre-LayerNorm** form, which is less sensitive to the learning-rate warm-up than the original
post-LayerNorm form (this repository still uses a short warm-up):

    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))          MLP = Linear(d, 4d), GELU, Linear(4d, d)

The additions are residual connections: each block adds a correction to the running
representation instead of replacing it. After the last block, a final LayerNorm and a linear
layer turn each position's vector into one score per character in the vocabulary, and a
softmax turns those into the probability of the next character. Training minimises the
cross-entropy between that distribution and the character that really came next.

## Decoding with a KV cache

When generating one character at a time, the keys and values of earlier positions never
change (the mask guarantees that). So the model keeps them in a cache and, for each new
character, computes only the new query, key and value, then attends over the cache. That
turns each step from work on the whole sequence into work on one position plus one pass over
the cached keys and values.
`test_kv_cache_matches_full_pass` checks that cached and uncached decoding give the same
result, and `results/full/quant.json` measures the speed difference.

## The cost

The score table is T x T, so time and memory grow with the square of the sequence length.
That is fine for the 128 characters used here and is the main reason long-context models
need other tricks (FlashAttention, sliding windows, and so on).
