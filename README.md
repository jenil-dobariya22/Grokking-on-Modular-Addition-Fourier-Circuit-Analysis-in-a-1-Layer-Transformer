# Grokking on Modular Addition: Fourier Circuit Analysis in a 1-Layer Transformer

A one-layer transformer is trained on `(a + b) mod 113` using only 30% of all input pairs and strong weight decay. The run reproduces **grokking** (Power et al., 2022): the model fits the training set by epoch 200, but test accuracy stays near chance for thousands of epochs and only crosses 90% at epoch 10,000. The notebook then opens the trained model up with TransformerLens and shows that it solves the task with a small set of **Fourier frequencies**, following Nanda et al. (2023).

For the full discussion, see [`writeup/`](writeup/).

## Key results

| Question | Result |
|---|---|
| Does the model grok? | Train acc 1.000 at epoch 200; test acc 0.026 at epoch 200 and 0.038 at epoch 1000; first > 90% at **epoch 10,000**; final test acc **0.9999** |
| Is the embedding sparse in the Fourier basis? | Yes. Key frequencies: **k ∈ {5, 9, 17, 28, 43}** |
| Does the unembedding use the same frequencies? | Yes. Overlap 5/5 |
| Are these frequencies causally needed? (logit ablation) | Keep only key freqs: test acc **1.000**. Remove them: test acc **0.003** |
| Is the embedding alone enough? (embedding projection) | Key freqs only: **0.997**. Non-key freqs only: **0.016** |
| When does the Fourier structure form? | Key-frequency energy fraction in the embedding: **0.141** (memorising checkpoint, epoch 500) → **0.851** (final model) |

## Setup

- **Task:** predict `(a + b) mod p`, `p = 113`. Input is the 3-token sequence `[a, b, =]`; the answer is read from the logits at the final `=` position.
- **Data:** all 113² = 12,769 pairs, shuffled with seed 0. 30% train (3,830), 70% test (8,939).
- **Model:** 1 layer, 4 heads (`d_head = 32`), `d_model = 128`, `d_mlp = 512`, ReLU, **no LayerNorm**, all biases frozen at zero. 227,313 parameters in total (226,048 trainable). Built with TransformerLens `HookedTransformer`.
- **Training:** AdamW, `lr = 1e-3`, `betas = (0.9, 0.98)`, `weight_decay = 1.0`. Full-batch, 12,000 epochs, cross-entropy in float64. Metrics logged every 100 epochs, checkpoint saved every 500 epochs.
- **Compute:** CPU only. No GPU needed.

## Repository structure

```
.
├── grokking_project.ipynb   # Main notebook: data, model, training, all analysis and figures
├── src/
│   ├── __init__.py
│   ├── data.py              # Dataset construction
│   ├── model.py             # Model definition
│   ├── train.py             # Training loop
│   └── analysis.py          # Fourier analysis and ablations
├── checkpoints/             # grokked.pt (final model), all_checkpoints.pt (every 500 epochs)
├── results/                 # training_history.pt (loss, accuracy, weight norm per 100 epochs)
├── figures/                 # Plots produced by the notebook
├── writeup/                 # Written report
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter notebook grokking_project.ipynb
```

Run all cells top to bottom. The notebook seeds Python, NumPy and PyTorch with `SEED = 0`, and the training run (12,000 full-batch epochs) is feasible on CPU. Running it will recreate `checkpoints/`, `results/` and `figures/`.

Environment used for the reported numbers: PyTorch 2.14.0 (CPU build), TransformerLens, conda env on Windows.

> Note: the notebook uses `HookedTransformer`, which prints a deprecation warning in recent TransformerLens versions. It still works; pin the TransformerLens version in `requirements.txt` if you want exact reproducibility.

## Notebook outline

1. Setup and reproducibility
2. Task and dataset (with assertions on every label)
3. Model
4. Training setup
5. Training
6. Training dynamics (accuracy, loss, log-scale loss, weight norm)
7. Fourier analysis of the embedding
8. MLP neurons and attention patterns
9. Unembedding spectrum
10. Causal test: Fourier ablation of output logits
11. Causal test: Fourier projection of the embedding
12. Memorising vs. grokked solution
13. Progress measure over training
14. Final results

## Figures

| File | Shows |
|---|---|
| `figures/grokking_accuracy.png` | Train vs. test accuracy |
| `figures/grokking_loss.png` | Train vs. test loss |
| `figures/grokking_loss_log_and_weight_norm.png` | Log-scale loss and total weight norm |
| `figures/embedding_fourier.png` | Embedding Fourier spectrum (grokked model) |
| `figures/embedding_fourier_memorizing.png` | Embedding Fourier spectrum (epoch 500) |
| `figures/unembedding_fourier.png` | Unembedding Fourier spectrum |
| `figures/neurons.png` | Top-4 MLP neurons over `(a, b)` and over `(a+b) mod p` |
| `figures/fourier_vs_accuracy.png` | Key-frequency energy fraction vs. test accuracy over training |

## Limitations

- One architecture, one seed, one train/test split. The specific key frequencies {5, 9, 17, 28, 43} will differ between runs.
- Key frequencies are picked with a simple threshold (mean + 2 std of the folded spectrum), not a tuned method.
- The analysis shows *that* the model uses a few Fourier frequencies and that they are causally necessary. It does not fully reverse-engineer the circuit (for example, it does not verify the trig-identity computation neuron by neuron).
- The role of weight decay is not isolated here: no run with different or zero weight decay is included.

## References

- Power, Burda, Edwards, Babuschkin, Misra (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.*
- Nanda, Chan, Lieberum, Smith, Steinhardt (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023.
- Nanda and Bloom (2022). *TransformerLens.* https://github.com/TransformerLensOrg/TransformerLens

## License

See [`LICENSE`](LICENSE).