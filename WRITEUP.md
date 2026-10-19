# Grokking on Modular Addition: Fourier Circuit Analysis in a 1-Layer Transformer

## Abstract

We train a one-layer, attention-and-MLP transformer on modular addition, `(a + b) mod 113`, using only 30% of all input pairs and AdamW with weight decay 1.0. The model reproduces grokking: it reaches 100% training accuracy by epoch 200, while test accuracy is still 2.6%. Test accuracy then stays low for thousands of epochs and first exceeds 90% at epoch 10,000, ending at 99.99%. Analysing the trained network, we find that its token embedding and unembedding are both sparse in the Fourier basis, concentrated on the same five frequencies, k ∈ {5, 9, 17, 28, 43}. Two independent interventions show these frequencies are causally necessary: keeping only them preserves test accuracy (1.000 on logits, 0.997 on the embedding), while removing them destroys it (0.003 and 0.016). Tracking a Fourier energy measure across checkpoints shows that this structure is already partly present at the memorising stage (0.141 of embedding energy at epoch 500) and grows gradually to 0.851, well before and during the sudden jump in test accuracy.

## 1. Introduction

Grokking (Power et al., 2022) is the phenomenon where a network first memorises its training set and only much later generalises to held-out data. Modular addition is a standard testbed. Nanda et al. (2023) showed that small transformers trained on it learn a Fourier-based algorithm: numbers are embedded as sinusoids at a few key frequencies, and the answer is computed with trigonometric identities.

This project reproduces that setting end to end and asks five concrete questions:

1. Does our setup show a clean memorise-then-generalise pattern?
2. Is the embedding sparse in the Fourier basis, and which frequencies matter?
3. Does the unembedding use the same frequencies?
4. Are these frequencies causally responsible for the model's accuracy?
5. When during training does the Fourier structure form?

## 2. Setup

### 2.1 Task and data

Given integers `a, b ∈ {0, …, 112}`, the model predicts `(a + b) mod p` with `p = 113`. Each example is the 3-token sequence `[a, b, =]`. The vocabulary has 114 tokens: the 113 numbers plus a special `=` token with id 113. The prediction is read from the logits at the final (`=`) position.

All 113² = 12,769 pairs are shuffled with a fixed seed. 30% (3,830 pairs) form the training set and 70% (8,939 pairs) the test set. The small training fraction is what makes memorisation easy and generalisation slow. The notebook asserts dataset sizes, token ranges, label ranges and every label against `(a + b) % p`.

### 2.2 Model

A deliberately small transformer, built with TransformerLens `HookedTransformer`:

| Hyperparameter | Value |
|---|---|
| Layers | 1 |
| Heads | 4 (`d_head = 32`) |
| `d_model` | 128 |
| `d_mlp` | 512 |
| Activation | ReLU |
| Normalisation | None |
| Biases | Frozen at zero |
| Parameters | 227,313 total, 226,048 trainable |

Removing LayerNorm and fixing all biases at zero means every computation goes through weight matrices, which keeps the learned circuit easier to analyse.

### 2.3 Training

- Optimiser: AdamW, `lr = 1e-3`, `betas = (0.9, 0.98)`, `weight_decay = 1.0`
- Loss: cross-entropy at the final position, computed in float64
- Full-batch, 12,000 epochs (one gradient step per epoch)
- Metrics logged every 100 epochs (train/test loss and accuracy, total parameter L2 norm); checkpoints saved every 500 epochs
- Seed 0 for Python, NumPy and PyTorch; CPU only

## 3. Results

### 3.1 Training dynamics

| Epoch | Train loss | Test loss | Train acc | Test acc |
|---|---|---|---|---|
| 0 | 4.734 | 4.733 | 0.007 | 0.009 |
| 100 | 2.825 | 7.508 | 0.329 | 0.009 |
| 200 | 0.030 | 19.03 | 1.000 | 0.026 |
| 1,000 | 0.000 | 28.49 | 1.000 | 0.038 |
| 5,000 | 0.000 | 21.86 | 1.000 | 0.075 |
| 8,000 | 0.000 | 9.694 | 1.000 | 0.264 |
| 10,000 | 0.000 | 0.293 | 1.000 | 0.927 |
| 11,900 | 0.000 | 0.0003 | 1.000 | 1.000 |

The model fits the training set by epoch 200 and then sits at near-chance test accuracy while test loss climbs above 30 (peak around epoch 1,400). Test accuracy then rises slowly and accelerates sharply from roughly epoch 8,000. The first logged epoch with test accuracy above 90% is **epoch 10,000** (logging resolution is 100 epochs). The final test accuracy is 0.9999.

The total parameter norm (log-scale plot in `figures/grokking_loss_log_and_weight_norm.png`) rises from about 34 at initialisation to a peak near 59.5 around epoch 900, then declines steadily to about 33 by the end of training. The decline in norm coincides with the period in which test loss falls. This is consistent with weight decay pushing the network from a high-norm memorising solution towards a lower-norm one, but this project does not run a weight-decay ablation, so we do not claim it as demonstrated.

### 3.2 The embedding is sparse in the Fourier basis

We take the DFT of the token embedding `W_E[:p]` along the token axis and compute the L2 norm of each frequency across the 128 embedding dimensions. The DC component is removed and conjugate frequencies `k` and `p − k` are folded together. Frequencies more than two standard deviations above the mean of the folded spectrum are called key frequencies.

The threshold is 29.58, and the key frequencies are **k ∈ {5, 9, 17, 28, 43}**: five of 56 non-DC frequencies.

### 3.3 The unembedding uses the same frequencies

Applying the same procedure to `W_U` gives exactly the same set, {5, 9, 17, 28, 43}, an overlap of 5/5. The model reads out its answer in the same Fourier basis it uses to write its inputs.

### 3.4 MLP neurons and attention

On the full 113 × 113 grid we cache MLP activations at the `=` position. The four highest-variance neurons (210, 324, 350, 502) are plotted over `(a, b)` and against `(a + b) mod p` in `figures/neurons.png`, where periodic structure is expected if neurons compute products of sinusoids.

Attention from the `=` query is nearly identical across all four heads: about 0.50 on `a`, 0.50 on `b` and 0.002 on `=` itself. The heads therefore act mainly as a mechanism that moves information about both operands into the final position, where the MLP combines them.

### 3.5 Causal tests

Correlation (a sparse spectrum) does not show the model *uses* those frequencies, so we intervene in two independent ways.

**Logit ablation.** For each example, we take the 113 output logits, apply a DFT along the answer axis, and either keep only the key frequencies (and conjugates) or keep only the non-key frequencies.

| | Full | Key only | Key removed |
|---|---|---|---|
| Train acc | 1.000 | 1.000 | 0.006 |
| Test acc | 1.000 | 1.000 | 0.003 |
| Test loss | 0.0003 | 0.0001 | 5.678 |

**Embedding projection.** We replace `W_E[:p]` with a version containing only the selected frequencies (plus the DC term), run the model, and measure test accuracy. The original embedding is always restored afterwards.

| Embedding keeps | Test acc |
|---|---|
| Key frequencies only | 0.997 |
| Non-key frequencies only | 0.016 (chance ≈ 0.009) |

Both interventions agree: five frequencies carry essentially all of the useful signal, and everything else is dispensable.

### 3.6 Memorising vs. grokked solution

We define the memorising checkpoint as the earliest saved checkpoint with train accuracy ≥ 99% and test accuracy < 10%. This is **epoch 500**. We measure the fraction of non-DC embedding Fourier energy that lies in the five key frequencies found in the final model:

| | Fraction of energy in key frequencies |
|---|---|
| Memorising checkpoint (epoch 500) | 0.141 |
| Grokked model | 0.851 |

For reference, a random embedding would place about 5/56 ≈ 0.09 of its energy in five frequency pairs, so the memorising checkpoint is only slightly above that baseline.

### 3.7 A progress measure

Computing the same energy fraction at every saved checkpoint (24 checkpoints, every 500 epochs) gives a continuous curve (`figures/fourier_vs_accuracy.png`). It starts near 0.09 at initialisation, reaches 0.14 at epoch 500, and then climbs smoothly to about 0.82 at the last checkpoint (epoch 11,500), while test accuracy sits near chance for most of that period and then jumps. The internal structure therefore changes gradually even though the test metric looks like a sudden transition. The final model's value of 0.851 is slightly higher than the epoch-11,500 checkpoint because training continued to epoch 12,000.

## 4. Discussion

The results fit the picture from Nanda et al. (2023). The network does not simply memorise 3,830 training pairs: over a long period, under strong weight decay, it builds a compact algorithm based on a handful of sinusoidal frequencies. That algorithm generalises to all 12,769 pairs, which is why test accuracy eventually reaches essentially 100%. The sharpness of grokking is then a property of the metric rather than of the underlying change: accuracy only registers once the generalising structure dominates the output, while the Fourier energy measure shows the structure growing steadily beforehand.

## 5. Limitations

- **Single run.** One architecture, one seed, one train/test split. The particular frequencies {5, 9, 17, 28, 43} are specific to this run, and the grokking epoch would vary with seed.
- **Heuristic frequency selection.** Key frequencies come from a mean + 2 std threshold on the folded spectrum. It gives a clean separation here, but it is not a principled test.
- **Incomplete circuit analysis.** We show the model uses these frequencies and that they are necessary and sufficient at the embedding and logit level. We do not verify the full computation, for example neuron-by-neuron trig-identity structure or the contribution of each head and neuron to each frequency.
- **Weight decay not isolated.** The weight-norm curve is consistent with a memorising-to-generalising transition driven by weight decay, but no run with different or zero weight decay is included.
- **Coarse logging.** Metrics are logged every 100 epochs and checkpoints saved every 500, so the grokking epoch and the memorising checkpoint are only resolved to that granularity.

## 6. Possible extensions

- Repeat over several seeds and report the spread in grokking epoch and key frequencies.
- Sweep the weight-decay coefficient and the training fraction.
- Decompose the output into per-neuron and per-head contributions per key frequency to verify the trig-identity circuit directly.
- Add restricted-loss and excluded-loss progress measures across checkpoints.

## 7. References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.*
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023.
- Nanda, N., Bloom, J. (2022). *TransformerLens.* https://github.com/TransformerLensOrg/TransformerLens