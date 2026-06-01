from pathlib import Path

src = Path("transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py")
dst = Path("transformer/scripts/train_idabd_hrtbda_v5_phase2_finetune.py")

if not src.exists():
    raise FileNotFoundError(f"Missing source v5 script: {src}")

text = src.read_text()

# Add CLI argument
if "--init-phase2-from" not in text:
    marker = "    return parser.parse_args()"
    if marker not in text:
        raise RuntimeError("Could not find parser return marker.")

    insert_arg = '''
    parser.add_argument(
        "--init-phase2-from",
        type=str,
        default=None,
        help="Optional xBD-trained Phase-II checkpoint used to initialize Phase II before IDA-BD fine-tuning.",
    )

'''
    text = text.replace(marker, insert_arg + marker)

# Add flexible Phase-II checkpoint loader
if "def load_phase2_init_weights_flexible" not in text:
    helper = r'''
def load_phase2_init_weights_flexible(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """
    Load an existing xBD-trained Phase-II checkpoint before fine-tuning on IDA-BD.
    """
    checkpoint_path = Path(checkpoint_path)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {checkpoint_path}")

    state = None
    for key in [
        "model",
        "model_state_dict",
        "state_dict",
        "phase2_model_state_dict",
        "phase2_state_dict",
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            print(f"Using Phase-II init checkpoint key: {key}", flush=True)
            break

    if state is None:
        if all(hasattr(v, "shape") for v in ckpt.values()):
            state = ckpt

    if state is None:
        print("Checkpoint keys:", sorted(list(ckpt.keys())), flush=True)
        raise RuntimeError("Could not find Phase-II model weights in checkpoint.")

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean_state[k[len("module."):]] = v
        else:
            clean_state[k] = v

    target_model = model.module if isinstance(model, nn.DataParallel) else model
    missing, unexpected = target_model.load_state_dict(clean_state, strict=False)

    print(f"Loaded Phase-II initialization checkpoint: {checkpoint_path}", flush=True)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}", flush=True)

    if missing:
        print("First missing keys:", missing[:10], flush=True)
    if unexpected:
        print("First unexpected keys:", unexpected[:10], flush=True)

'''
    idx = text.find("\n# -----------------------------\n# Training")
    if idx == -1:
        idx = text.find("\ndef train_phase2")
    if idx == -1:
        raise RuntimeError("Could not find insertion point before train_phase2.")

    text = text[:idx] + helper + text[idx:]

# Insert initialization inside train_phase2 before DataParallel
if "Applying external Phase-II initialization before fine-tuning" not in text:
    train_idx = text.find("def train_phase2")
    if train_idx == -1:
        raise RuntimeError("Could not find def train_phase2.")

    dp_idx = text.find("if torch.cuda.device_count() > 1", train_idx)
    if dp_idx == -1:
        raise RuntimeError("Could not find DataParallel block inside train_phase2.")

    insert_init = '''
    if getattr(args, "init_phase2_from", None):
        print("Applying external Phase-II initialization before fine-tuning.", flush=True)
        load_phase2_init_weights_flexible(
            model=model,
            checkpoint_path=Path(args.init_phase2_from),
            device=device,
        )

'''
    text = text[:dp_idx] + insert_init + text[dp_idx:]

dst.write_text(text)
print(f"Wrote real fine-tuning script: {dst}")