

# Ablation Results

Causal verification that the 4 key frequencies (`[1, 2, 17, 28]`) found in the embedding and unembedding Fourier spectra are actually responsible for the model's predictions — not just correlated with them.

## Logit-space ablation (Step 22)

Restricting the final logits to only the key frequencies vs. removing them:

TRAIN
Full | loss = 0.0000 | acc = 1.000
Key only | loss = 0.0000 | acc = 1.000
Key removed| loss = 5.3449 | acc = 0.018

TEST
Full | loss = 0.0000 | acc = 1.000
Key only | loss = 0.0000 | acc = 1.000
Key removed| loss = 5.5351 | acc = 0.012



**Interpretation:** keeping only the key frequencies is indistinguishable from the full model (loss and accuracy identical). Removing them collapses test accuracy to 0.012 — close to the 1/113 ≈ 0.0088 chance level.

## Embedding-space ablation (Step 23)

Projecting the embedding matrix onto only the key frequencies vs. only the non-key frequencies, then evaluating test accuracy:

Keep only key frequencies : 1.0
Keep only NON-key frequencies : 0.009732631966471672


**Interpretation:** the embedding alone — reduced to just 4 frequencies — is sufficient to fully reconstruct the circuit's behavior (100% test accuracy). The remaining ~106 non-key frequencies carry essentially no useful signal (accuracy ≈ chance).

## Memorizing vs. grokked — Fourier energy fraction (Step 24)

Fraction of total embedding Fourier energy concentrated in the 4 key frequencies, compared between the memorizing checkpoint (epoch 500) and the final grokked model:

Memorizing (epoch 500) : 0.118
Grokked (final model) : 0.889


**Interpretation:** during memorization, the embedding's energy is spread across many frequencies (only 11.8% in the eventual key set). By the end of training, 88.9% of all energy is concentrated in just 4 frequencies — the "cleanup" phase of grokking.

## Key frequencies: embedding vs. unembedding

Embedding key frequencies : [1, 2, 17, 28]
Unembedding key frequencies: [1, 2, 17, 28]
Overlap : [1, 2, 17, 28] (100%)


The embedding and unembedding independently converge on the exact same four frequencies — strong evidence that both ends of the network are using the same shared Fourier basis to encode and decode the answer.