# AMP Challenge 2027 — DiT-AMP submission

Discrete denoising diffusion (D3PM) with a DiT backbone for antimicrobial peptide design.

## Quick start

```bash
uv sync
uv run generate
```

This writes:

```
generate/
  library.fasta   # 50,000 unique AMP designs
  top.fasta       # top-100 ranked candidates
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-sequences` | `50000` | Library size |
| `--top-k` | `100` | Ranked shortlist size |
| `--seed` | `42` | RNG seed. The same seed reproduces `library.fasta` and `top.fasta`. |
| `--device` | cuda if available | Torch device used for sampling |

Every run samples from `checkpoint/AMP-DiT.pth`. The full 50k library needs a GPU:

```bash
uv run generate --n-sequences 8   # smoke test
```

## Method summary

- **Model:** D3PM over a 28-token amino-acid alphabet with a Llama-style DiT denoiser (`dim=512`, `4` layers, `8` heads, `1000` diffusion steps). Each batch is sampled at one length drawn uniformly from 8 to 30.
- **Conditioning:** two scalars (MACREL AMP / hemolysis probabilities). Sampling conditions on high AMP and high hemolysis (`1.0, 1.0`).
- **Training data:** antibacterial peptides from public AMP resources (see `data/` and the DiT-AMP project).
- **Ranking:** MACREL AMP − hemolysis score, then reject any top candidate with >80% Levenshtein identity to `data/antibacterial.fasta`.

## Verify

```bash
uv run python scripts/verify_submission.py https://github.com/Alirzeanoroozi/amp-challenge-2027
```

## License

BSD-3-Clause — free to use, modify, and redistribute. See [LICENSE](LICENSE).


## Data

We use the Therapeutic Peptides dataset (Xiao et al., 2025)
for training and evaluation. We retain only peptides com-
posed of the 20 canonical amino acids (|V| = 20) with
length between 5 and 30 residues.

Xiao, B., Zhou, Y., Zhao, L., Huang, H., Fei, X., and
Zhang, Y.-B. A comprehensive dataset of therapeu-
tic peptides on multi-function property and structure
information. Scientific Data, 12(1):1213, 2025. doi:
10.1038/s41597-025-05528-1. URL https://doi.
org/10.1038/s41597-025-05528-1.