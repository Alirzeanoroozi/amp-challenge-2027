import argparse
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

# Must be set before torch is imported if cuBLAS is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parent
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50

# Architecture of checkpoint/AMP-DiT.pth (DiT-AMP configs/config.yaml).
N = 28
N_T = 1000
HYBRID_LOSS_COEFF = 0.001
DIM = 512
N_LAYERS = 4
N_HEADS = 8
MIN_SAMPLE_LENGTH = 8
MAX_SAMPLE_LENGTH = 30
BATCH_SIZE = 128
# MACREL probabilities in [0, 1]. 1.0 is high AMP and high hemolysis.
COND_AMP = 1.0
COND_HEMO = 1.0
CHECKPOINT = REPO_ROOT / "checkpoint" / "AMP-DiT.pth"


def tokens_to_sequence(tokens: list[int]) -> str:
    sent = "".join(chr(token + ord("A") - 1) for token in tokens)
    return sent.split("@", 1)[0]


def is_valid_sequence(seq: str) -> bool:
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and set(seq) <= STANDARD_AMINO_ACIDS


def _write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def _read_fasta(path: Path) -> list[str]:
    sequences: list[str] = []
    parts: list[str] = []
    header = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequences.append("".join(parts).upper())
            header, parts = line[1:], []
        else:
            parts.append(line)
    if header is not None:
        sequences.append("".join(parts).upper())
    return sequences


def _ensure_import_shims() -> None:
    """Let DiT-AMP ``d3pm.py`` import without its training-only dependencies."""
    if "omegaconf" not in sys.modules:
        try:
            import omegaconf  # noqa: F401
        except ModuleNotFoundError:
            omegaconf = types.ModuleType("omegaconf")

            class DictConfig(dict):
                pass

            class OmegaConf:
                @staticmethod
                def to_container(cfg, resolve=False):
                    return dict(cfg)

            omegaconf.DictConfig = DictConfig
            omegaconf.OmegaConf = OmegaConf
            sys.modules["omegaconf"] = omegaconf

    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError:
            transformers = types.ModuleType("transformers")

            def get_scheduler(*args, **kwargs):
                raise RuntimeError("get_scheduler is only used for training")

            transformers.get_scheduler = get_scheduler
            sys.modules["transformers"] = transformers

    if "pytorch_lightning" not in sys.modules:
        try:
            import pytorch_lightning  # noqa: F401
        except ModuleNotFoundError:
            pl = types.ModuleType("pytorch_lightning")

            class LightningModule:
                def __init__(self, *args, **kwargs):
                    pass

                def save_hyperparameters(self, *args, **kwargs):
                    pass

            pl.LightningModule = LightningModule
            sys.modules["pytorch_lightning"] = pl


def _import_model_api():
    """Import ``D3PM`` and ``DDiT_Llama`` the way DiT-AMP defines them."""
    pkg_dir = str(PKG_DIR)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    _ensure_import_shims()
    from d3pm import D3PM
    from dit import DDiT_Llama

    return D3PM, DDiT_Llama


def _seed_everything(seed: int, device_type: str) -> None:
    import torch

    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for name, enabled in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_cudnn_sdp", False),
            ("enable_math_sdp", True),
        ):
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                fn(enabled)


def load_model(device):
    import torch

    D3PM, DDiT_Llama = _import_model_api()
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT} (cwd={os.getcwd()})"
        )
    x0_model = DDiT_Llama(N=N, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS)
    d3pm = D3PM(
        x0_model,
        n_T=N_T,
        num_classes=N,
        hybrid_loss_coeff=HYBRID_LOSS_COEFF,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("d3pm.") for key in state):
        state = {key.removeprefix("d3pm."): value for key, value in state.items()}
    d3pm.load_state_dict(state)
    d3pm.to(device)
    d3pm.eval()
    return d3pm


def sample_live(
    n_sequences: int,
    *,
    seed: int,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    """Draw ``n_sequences`` valid, unique peptides from ``D3PM.sample``."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    d3pm = load_model(device_t)
    # Reseed after module construction so weight init does not consume the
    # stream that ``D3PM.sample`` reads via ``torch.rand``.
    _seed_everything(seed, device_t.type)

    antibacterial = set(_read_fasta(REPO_ROOT / "data" / "antibacterial.fasta"))
    sequences: list[str] = []
    seen: set[str] = set()
    drawn = 0
    max_draws = max(n_sequences * 30, batch_size)
    batches = 0

    while len(sequences) < n_sequences:
        if drawn >= max_draws:
            raise RuntimeError(
                f"Only collected {len(sequences)}/{n_sequences} valid unique sequences "
                f"after {drawn} draws."
            )
        remaining = n_sequences - len(sequences)
        current = batch_size if remaining > batch_size else remaining
        # One length per batch, drawn from the seeded torch stream.
        length = int(
            torch.randint(
                MIN_SAMPLE_LENGTH,
                MAX_SAMPLE_LENGTH + 1,
                (),
                device=device_t,
            ).item()
        )
        init_noise = torch.randint(
            0,
            N,
            (current, length),
            device=device_t,
        )
        cond = torch.tensor(
            [COND_AMP, COND_HEMO],
            device=device_t,
            dtype=torch.float32,
        ).unsqueeze(0).expand(current, -1)
        outputs = d3pm.sample(init_noise, cond=cond)
        drawn += current
        batches += 1
        for i in range(current):
            seq = tokens_to_sequence(outputs[i].detach().cpu().tolist())
            if not is_valid_sequence(seq):
                continue
            if seq in seen or seq in antibacterial:
                continue
            seen.add(seq)
            sequences.append(seq)
            if len(sequences) >= n_sequences:
                break
        print(
            f"Sampled batch {batches} (length {length}): "
            f"{len(sequences)}/{n_sequences} kept ({drawn} draws)",
            flush=True,
        )

    return sequences[:n_sequences]


def macrel_rank_scores(sequences: list[str]) -> list[tuple[str, float]]:
    """Score finished samples with MACREL. Higher is better: AMP − hemolysis."""
    pkg_dir = str(PKG_DIR)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    from macrel.run_macrel import run_macrel

    scored: list[tuple[str, float]] = []
    total = len(sequences)
    for i, seq in enumerate(sequences, start=1):
        try:
            amp, hemo = run_macrel(seq)
            score = float(amp) - float(hemo)
        except Exception:
            score = -2.0
        scored.append((seq, score))
        if i % 500 == 0 or i == total:
            print(f"MACREL scored {i}/{total}", flush=True)
    return scored


def select_top_k(
    scored: list[tuple[str, float]],
    top_k: int,
    references: set[str],
    max_similarity: float = 0.8,
) -> list[str]:
    import Levenshtein

    refs_by_len: dict[int, list[str]] = defaultdict(list)
    for ref in references:
        refs_by_len[len(ref)].append(ref)

    def too_similar(seq: str) -> bool:
        length = len(seq)
        lo = max(MIN_LENGTH, int(length * 0.7))
        hi = min(MAX_LENGTH, int(length / 0.7) + 1)
        for ref_len in range(lo, hi + 1):
            for ref in refs_by_len.get(ref_len, []):
                if Levenshtein.ratio(seq, ref) > max_similarity:
                    return True
        return False

    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    selected: list[str] = []
    for seq, _score in ranked:
        if too_similar(seq):
            continue
        selected.append(seq)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        raise RuntimeError(
            f"Only found {len(selected)}/{top_k} sequences below {max_similarity} "
            "identity to the antibacterial reference set."
        )
    return selected


def generate_live(
    n_sequences: int,
    top_k: int,
    seed: int,
    device: str | None = None,
) -> tuple[list[str], list[str]]:
    sequences = sample_live(n_sequences, seed=seed, device=device)
    antibacterial = set(_read_fasta(REPO_ROOT / "data" / "antibacterial.fasta"))
    scored = macrel_rank_scores(sequences)
    top = select_top_k(scored, top_k, antibacterial)
    return sequences, top


def main() -> None:
    entry_point = Path(sys.argv[0]).stem
    parser = argparse.ArgumentParser(description="DiT-AMP AMP Challenge generator")
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    out_dir = Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)

    library, top = generate_live(
        args.n_sequences,
        args.top_k,
        args.seed,
        device=args.device,
    )
    _write_fasta(library, out_dir / "library.fasta")
    _write_fasta(top, out_dir / "top.fasta")

    print(f"Generated {len(library)} sequences → {out_dir / 'library.fasta'}")
    print(f"Top {len(top)} sequences → {out_dir / 'top.fasta'}")


if __name__ == "__main__":
    main()
