"""Seed Lightning, NumPy, and worker RNGs, then run NeMo's official trainer."""

import os
import runpy

from lightning.pytorch import seed_everything


seed = int(os.environ.get("TRAIN_SEED", "42"))
entrypoint = os.environ.get(
    "NEMO_FINETUNE_ENTRYPOINT",
    "/nemo-src/examples/asr/speech_to_text_finetune.py",
)

seed_everything(seed, workers=True)
print(f"training_seed={seed}", flush=True)
runpy.run_path(entrypoint, run_name="__main__")
