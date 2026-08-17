import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import nibabel as nib
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_name(raw: str, index: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    cleaned = cleaned.strip("_-") or f"sample_{index:02d}"
    return f"{index:02d}_{cleaned}"


class SegmentationVisualizationCallback:
    """Visualize and persist segmentation checkpoints for 3D volumes."""

    def __init__(
        self,
        save_root: Path,
        slices: Optional[Sequence[int]] = None,
        num_slices: int = 3,
        max_samples: int = 2,
        overlay_alpha: float = 0.45,
        sampler: str = 'categorical',
        sampling_steps: Optional[int] = None,
        final_step_mode: Optional[str] = None,
    ) -> None:
        self.save_root = Path(save_root)
        self.user_slices = list(slices) if slices is not None else None
        self.num_slices = max(1, num_slices)
        self.max_samples = max(1, max_samples)
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.05, 0.95))
        self.sampler = str(sampler).lower()
        self.sampling_steps = sampling_steps
        self.final_step_mode = final_step_mode

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _save_nifti(volume: np.ndarray, affine: np.ndarray, destination: Path, dtype=np.float32) -> None:
        volume_np = np.asarray(volume, dtype=dtype)
        affine_np = np.asarray(affine, dtype=np.float32)
        if affine_np.shape != (4, 4):
            affine_np = np.eye(4, dtype=np.float32)
        nib.save(nib.Nifti1Image(volume_np, affine_np), str(destination))

    def _determine_slices(self, depth: int, gt_seg: np.ndarray = None) -> List[int]:
        if depth <= 0:
            return [0]

        if self.user_slices:
            return [int(np.clip(idx, 0, depth - 1)) for idx in self.user_slices]

        if gt_seg is not None and self.num_slices > 0:
            slice_scores = []
            for i in range(depth):
                unique_labels = len(np.unique(gt_seg[i]))
                slice_scores.append((i, unique_labels))
            slice_scores.sort(key=lambda x: x[1], reverse=True)
            selected_slices = [idx for idx, _ in slice_scores[:self.num_slices]]
            return sorted(selected_slices)

        if self.num_slices == 1:
            return [depth // 2]

        positions = np.linspace(0, depth - 1, num=self.num_slices)
        indices = sorted({int(round(pos)) for pos in positions})
        return [int(np.clip(idx, 0, depth - 1)) for idx in indices]

    @staticmethod
    def _normalise_image(image: np.ndarray) -> np.ndarray:
        vmin, vmax = image.min(), image.max()
        if math.isclose(vmin, vmax):
            return np.zeros_like(image)
        return (image - vmin) / (vmax - vmin)

    def _build_cmap(self, num_classes: int) -> ListedColormap:
        base_cmap = plt.get_cmap("tab20")
        colors = [base_cmap(i % base_cmap.N) for i in range(num_classes + 1)]
        colors[0] = (0.0, 0.0, 0.0, 0.0)
        return ListedColormap(colors)

    def _render_overview(
        self,
        fig_path: Path,
        image_volume: np.ndarray,
        pred_seg: np.ndarray,
        gt_seg: np.ndarray,
        slices: Iterable[int],
        cmap: ListedColormap,
        num_classes: int,
    ) -> None:
        slices = list(slices)
        fig, axes = plt.subplots(len(slices), 3, figsize=(12, 4 * len(slices)))
        axes = np.atleast_2d(axes)

        for row, sl in enumerate(slices):
            img_slice = self._normalise_image(image_volume[sl])
            pred_slice = pred_seg[sl]
            gt_slice = gt_seg[sl]

            axes[row, 0].imshow(img_slice, cmap="gray")
            axes[row, 0].set_title(f"Slice {sl} – Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(img_slice, cmap="gray")
            axes[row, 1].imshow(pred_slice, cmap=cmap, alpha=self.overlay_alpha, vmin=0, vmax=num_classes)
            axes[row, 1].set_title("Prediction")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_slice, cmap="gray")
            axes[row, 2].imshow(gt_slice, cmap=cmap, alpha=self.overlay_alpha, vmin=0, vmax=num_classes)
            axes[row, 2].set_title("Ground Truth")
            axes[row, 2].axis("off")

        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

    def _render_per_class(
        self,
        fig_path: Path,
        image_volume: np.ndarray,
        pred_seg: np.ndarray,
        gt_seg: np.ndarray,
        slices: Iterable[int],
        classes: Sequence[int],
    ) -> None:
        slices = list(slices)
        num_classes = len(classes)
        fig, axes = plt.subplots(
            len(slices),
            num_classes * 2,
            figsize=(4 * num_classes * 2, 4 * len(slices))
        )
        axes = np.atleast_2d(axes)

        for row, sl in enumerate(slices):
            img_slice = self._normalise_image(image_volume[sl])
            for col, class_id in enumerate(classes):
                pred_mask = (pred_seg[sl] == class_id).astype(float)
                gt_mask = (gt_seg[sl] == class_id).astype(float)

                ax_pred = axes[row, col * 2]
                ax_gt = axes[row, col * 2 + 1]

                ax_pred.imshow(img_slice, cmap="gray")
                ax_pred.imshow(pred_mask, cmap="Reds", alpha=self.overlay_alpha)
                ax_pred.set_title(f"Slice {sl} – Class {class_id} Pred")
                ax_pred.axis("off")

                ax_gt.imshow(img_slice, cmap="gray")
                ax_gt.imshow(gt_mask, cmap="Blues", alpha=self.overlay_alpha)
                ax_gt.set_title(f"Slice {sl} – Class {class_id} GT")
                ax_gt.axis("off")

        fig.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

    def on_checkpoint(
        self,
        ema_model: torch.nn.Module,
        batch: dict,
        milestone,
        results_folder: str,
        reason: str = "step",
    ) -> None:
        save_dir = Path(results_folder)
        milestone_dir = save_dir / "visualizations" / f"milestone_{milestone}"
        _ensure_dir(milestone_dir)

        device = next(ema_model.parameters()).device
        ema_model_was_training = ema_model.training
        ema_model.eval()

        images = batch.get('image')
        gt_segmentations = batch.get('segmentation')
        names = batch.get('name')
        affines = batch.get('affine')
        preds_seg = batch.get('prediction')

        if images is None or gt_segmentations is None:
            return

        if images.ndim == 4:
            images = images.unsqueeze(2)
        if gt_segmentations.ndim == 4:
            gt_segmentations = gt_segmentations.unsqueeze(2)

        if preds_seg is None:
            if not hasattr(ema_model, 'sample') or not callable(ema_model.sample):
                raise TypeError(f"{type(ema_model).__name__} does not provide a sample() method")
            with torch.no_grad():
                auto_sample = ema_model.sample(
                    batch_size=images.shape[0],
                    proc=False,
                    channel_cond=images.to(device),
                    sampler=self.sampler,
                    num_steps=self.sampling_steps,
                    final_step_mode=self.final_step_mode,
                ).cpu()

            if auto_sample.ndim == 4:
                auto_sample = auto_sample.unsqueeze(2)

            preds_seg = auto_sample

        if ema_model_was_training:
            ema_model.train()

        num_samples = min(self.max_samples, images.shape[0])

        for idx in range(num_samples):
            image_volume = self._tensor_to_numpy(images[idx])[0]
            gt_seg = self._tensor_to_numpy(gt_segmentations[idx]).squeeze(0)

            pred_numpy = self._tensor_to_numpy(preds_seg[idx])
            if pred_numpy.shape[0] == 1:
                pred_seg = pred_numpy.squeeze(0)
            elif pred_numpy.ndim == 4:
                pred_seg = np.argmax(pred_numpy, axis=0)
            else:
                pred_seg = pred_numpy

            classes = sorted({int(c) for c in np.unique(np.concatenate((pred_seg.flatten(), gt_seg.flatten()))) if c != 0})
            num_classes = max(classes) if classes else 0
            cmap = self._build_cmap(max(num_classes, 1))
            slice_indices = self._determine_slices(image_volume.shape[0], gt_seg)

            sample_name = _sanitize_name(names[idx] if names is not None else f"sample_{idx}", idx)
            sample_dir = milestone_dir / sample_name
            _ensure_dir(sample_dir)

            affine = np.eye(4, dtype=np.float32)
            if affines is not None:
                affine = self._tensor_to_numpy(affines[idx])

            self._save_nifti(image_volume, affine, sample_dir / "image.nii.gz", dtype=np.float32)
            self._save_nifti(pred_seg, affine, sample_dir / "prediction.nii.gz", dtype=np.int16)
            self._save_nifti(gt_seg, affine, sample_dir / "ground_truth.nii.gz", dtype=np.int16)

            self._render_overview(
                fig_path=sample_dir / f"overview_{reason}.png",
                image_volume=image_volume,
                pred_seg=pred_seg,
                gt_seg=gt_seg,
                slices=slice_indices,
                cmap=cmap,
                num_classes=max(num_classes, 1)
            )

            if classes:
                self._render_per_class(
                    fig_path=sample_dir / f"per_class_{reason}.png",
                    image_volume=image_volume,
                    pred_seg=pred_seg,
                    gt_seg=gt_seg,
                    slices=slice_indices,
                    classes=classes,
                )
