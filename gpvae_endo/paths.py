"""Where a run reads and writes files. All folders come from the run's config file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def weights_dir() -> Path:
    """Folder with the pretrained weights (DINOv2 / GastroNet ViT, DUCKNet).

    ``weights/`` at the repository root. The original code kept them in ``Models/Weights/``; that folder is
    still used when it exists and ``weights/`` does not, so an existing download keeps working.
    """
    new, old = REPO_ROOT / "weights", REPO_ROOT / "Models" / "Weights"
    if not new.exists() and old.exists():
        return old
    return new


@dataclass(frozen=True)
class OutputPaths:
    """Output folders of one run (same names and layout as the original driver)."""

    output: str             # FOLDER_OUTPUT: plots ("Pics/")
    logs: str               # FOLDER_LOGS: run logs ("Logs/")
    models: str             # FOLDER_TO_SAVE_MODELS: trained models and checkpoints ("Saved_Models/")
    recon_dataset: str      # reconstructed frames and videos of this video
    grids: str              # per-frame comparison grids (only with MUST_SAVE_RECONSTRUCTION_GRIDS)
    frames: str             # per-frame reconstructions
    debug_dataset: str      # side-by-side comparison images ("Debug/<video>/")
    reconstructed_root: str  # FOLDER_TO_SAVE_RECONSTRUCTED
    save_grids: bool

    @classmethod
    def from_config(cls, cfg) -> "OutputPaths":
        dataset_name = cfg.substring_filter.strip().replace("\\", "_").replace("/", "_")
        recon_dataset = os.path.join(cfg.folder_to_save_reconstructed, dataset_name)
        return cls(
            output=cfg.folder_output,
            logs=cfg.folder_logs,
            models=cfg.folder_to_save_models,
            recon_dataset=recon_dataset,
            grids=os.path.join(recon_dataset, "Grids"),
            frames=os.path.join(recon_dataset, "Frames"),
            debug_dataset=os.path.join(cfg.folder_to_save_frames_for_debugging, dataset_name),
            reconstructed_root=cfg.folder_to_save_reconstructed,
            save_grids=bool(cfg.must_save_reconstruction_grids),
        )

    def create(self) -> None:
        """Create the output folders the original driver created at start-up, and also the logs, plots and
        models folders: the original expected those to exist already (they were in its own folder), so a run
        from a fresh clone or an empty run folder failed at its first log line."""
        for folder in (self.logs, self.output, self.models):
            os.makedirs(folder, exist_ok=True)
        os.makedirs(self.recon_dataset, exist_ok=True)
        if self.save_grids:
            os.makedirs(self.grids, exist_ok=True)
        os.makedirs(self.frames, exist_ok=True)
        os.makedirs(self.debug_dataset, exist_ok=True)
