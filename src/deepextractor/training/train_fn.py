import torch
import torch.nn as nn
from tqdm import tqdm

from deepextractor.utils.checkpoints import load_checkpoint, load_optimizer, save_checkpoint
from deepextractor.utils.io import check_accuracy, get_loaders
from deepextractor.utils.visualization import save_predictions_as_plots


def train_fn(loader, model, model_name, optimizer, loss_fn, scaler, device):
    """Train the model for one epoch and return average losses."""
    loop = tqdm(loader, desc="Training on batch")

    epoch_loss = 0
    epoch_noise_loss = 0
    epoch_constraint_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.amp.autocast(autocast_device):
            predictions = model(data)

            if model_name == "UNET1D_diff":
                noise_pred = predictions[:, 0:1, :]
                residual_pred = predictions[:, 1:2, :]
                reconstructed = noise_pred + residual_pred
                constraint_loss = loss_fn(reconstructed, data)
                noise_loss = loss_fn(noise_pred, targets)
                loss = constraint_loss + noise_loss
                epoch_noise_loss += noise_loss.item()
                epoch_constraint_loss += constraint_loss.item()
            else:
                loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        if model_name == "UNET1D_diff":
            loop.set_postfix(
                total_loss=loss.item(),
                constraint_loss=constraint_loss.item(),
                noise_loss=noise_loss.item(),
            )
        else:
            loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(loader)
    avg_noise_loss = epoch_noise_loss / len(loader) if model_name == "UNET1D_diff" else 0
    avg_constraint_loss = (
        epoch_constraint_loss / len(loader) if model_name == "UNET1D_diff" else 0
    )

    return avg_loss, avg_noise_loss, avg_constraint_loss


def train_fn_td(
    loader,
    model,
    optimizer,
    loss_fn,
    scaler,
    device,
    *,
    residual_channels: bool = False,
    residual_weight: float = 1.0,
    residual_mode: str = "true",
    use_amp: bool = False,
):
    """Train the two-detector time-domain separation model for one epoch.

    Expects the DataLoader to yield:
        data    — (B, 2, T)  H1+L1 strain (standard-scaled inputs)
        targets — (B, 4, T)  [bg_H1, bg_L1, sig_H1, sig_L1] (whitened)

    Args:
        residual_channels: If True, model outputs 6 channels; the extra 2 are
            a residual term that enforces input reconstruction.
        residual_weight: Weight applied to the residual loss term.
        residual_mode: How the residual loss is computed —
            "true"       : residual target = data - (tgt_bg + tgt_sig)
            "sum"        : loss on (pred_bg + pred_sig + pred_res) vs data
            "sum_detach" : same but bg+sig gradients are detached
        use_amp: Enable mixed-precision autocast. Disabled by default —
            Snellius training found AMP unstable with whitened targets.

    Returns:
        (avg_total, avg_bg, avg_sig)           if residual_channels=False
        (avg_total, avg_bg, avg_sig, avg_res)  if residual_channels=True
    """
    loop = tqdm(loader, desc="Training on batch")
    tot = bg_acc = sig_acc = res_acc = 0.0

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for data, targets in loop:
        data = data.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(autocast_device, enabled=use_amp):
            preds = model(data)

            if residual_channels:
                if preds.shape[1] != 6:
                    raise ValueError(f"residual_channels=True requires 6 output channels, got {preds.shape[1]}")
                pred_bg  = preds[:, 0:2]
                pred_sig = preds[:, 2:4]
                pred_res = preds[:, 4:6]
            else:
                if preds.shape[1] != 4:
                    raise ValueError(f"residual_channels=False requires 4 output channels, got {preds.shape[1]}")
                pred_bg  = preds[:, 0:2]
                pred_sig = preds[:, 2:4]
                pred_res = None

            tgt_bg  = targets[:, 0:2]
            tgt_sig = targets[:, 2:4]

            bg_loss  = loss_fn(pred_bg,  tgt_bg)
            sig_loss = loss_fn(pred_sig, tgt_sig)

            if residual_channels:
                if residual_mode == "true":
                    res_tgt  = data - (tgt_bg + tgt_sig)
                    res_loss = loss_fn(pred_res, res_tgt)
                elif residual_mode == "sum":
                    res_loss = loss_fn(pred_bg + pred_sig + pred_res, data)
                elif residual_mode == "sum_detach":
                    res_loss = loss_fn((pred_bg + pred_sig).detach() + pred_res, data)
                else:
                    raise ValueError(f"Unknown residual_mode '{residual_mode}'. Choose 'true', 'sum', or 'sum_detach'.")
                loss = 0.5 * (bg_loss + sig_loss) + residual_weight * res_loss
            else:
                res_loss = None
                loss = 0.5 * (bg_loss + sig_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot     += loss.item()
        bg_acc  += bg_loss.item()
        sig_acc += sig_loss.item()
        if res_loss is not None:
            res_acc += res_loss.item()

        postfix = {"total": loss.item(), "bg": bg_loss.item(), "sig": sig_loss.item()}
        if res_loss is not None:
            postfix["res"] = res_loss.item()
        loop.set_postfix(**postfix)

    n = max(1, len(loader))
    if residual_channels:
        return tot / n, bg_acc / n, sig_acc / n, res_acc / n
    return tot / n, bg_acc / n, sig_acc / n


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss normalised only over active (mask=1) elements.

    Args:
        pred, target: (B, C, T).
        mask: (B, C) or (C,) — 1.0 for channels that should contribute to the
            loss, 0.0 for channels belonging to an offline/excluded detector
            (broadcast over T). Excluded channels get exactly zero gradient,
            rather than being trained toward an arbitrary zero target.
    """
    mask_bc = mask.unsqueeze(-1) if mask.dim() == 2 else mask.view(1, -1, 1)
    sq_err = (pred - target) ** 2 * mask_bc
    n_active = mask_bc.expand_as(sq_err).sum()
    if n_active == 0:
        raise ValueError("masked_mse_loss: mask excludes every element in this batch.")
    return sq_err.sum() / n_active


def _masked_mse_components(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Split a combined masked-MSE loss into (total, background, signal).

    Assumes the standard channel layout HDF5SeparationDataset produces when
    ``target_signal_only=False``: the first half of channels are background
    targets, the second half signal targets, both in the same detector
    order. ``total`` here is mathematically identical to
    ``masked_mse_loss(pred, target, mask)`` on the full tensor -- background
    and signal always have equal active-element counts (the presence mask
    applies identically to both), so ``0.5 * (bg + sig)`` and the combined
    MSE over the whole tensor are the same value. This is purely additional
    diagnostic detail, not a change to what gets optimized.
    """
    half = pred.shape[1] // 2
    if mask.dim() == 2:
        mask_bg, mask_sig = mask[:, :half], mask[:, half:]
    else:
        mask_bg, mask_sig = mask[:half], mask[half:]
    bg_loss = masked_mse_loss(pred[:, :half], target[:, :half], mask_bg)
    sig_loss = masked_mse_loss(pred[:, half:], target[:, half:], mask_sig)
    total = 0.5 * (bg_loss + sig_loss)
    return total, bg_loss, sig_loss


def train_fn_separation(loader, model, optimizer, scaler, device, use_amp=False):
    """Train the multi-detector separation model for one epoch with a masked loss.

    Expects the DataLoader to yield ``(data, targets, mask)`` — the 3-tuple
    produced by :class:`~deepextractor.data.HDF5SeparationDataset` when
    constructed with ``active_detectors`` set:

        data    — (B, 2*n_det, T)  scaled strain + presence-flag channels
        targets — (B, 2*n_det, T)  [bg_det1..detN, sig_det1..detN]
        mask    — (B, 2*n_det) or (2*n_det,) — 1.0 active / 0.0 offline

    Unlike :func:`train_fn_td`, this isn't hardcoded to 2 detectors — it
    works for any ``n_det`` since the masked loss is computed over whatever
    channel width the batch actually has.

    Args:
        scaler: ``torch.cuda.amp.GradScaler`` (same role as in ``train_fn_td``).
        use_amp: Enable mixed-precision autocast. Off by default — prior TD
            training on Snellius found AMP unstable with whitened targets.

    Returns:
        (avg_total, avg_bg, avg_sig) over the epoch (mean over batches).
        Backward pass uses avg_total only, same value as before this
        component split was added.
    """
    loop = tqdm(loader, desc="Training on batch")
    tot = bg_acc = sig_acc = 0.0
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for data, targets, mask in loop:
        data = data.to(device)
        targets = targets.float().to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(autocast_device, enabled=use_amp):
            preds = model(data)
            loss, bg_loss, sig_loss = _masked_mse_components(preds, targets, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot += loss.item()
        bg_acc += bg_loss.item()
        sig_acc += sig_loss.item()
        loop.set_postfix(total=loss.item(), bg=bg_loss.item(), sig=sig_loss.item())

    n = max(1, len(loader))
    return tot / n, bg_acc / n, sig_acc / n


@torch.no_grad()
def eval_fn_separation(loader, model, device):
    """Evaluate the multi-detector separation model with a masked loss.

    Mirrors :func:`train_fn_separation` — see its docstring for the expected
    ``(data, targets, mask)`` loader output and the ``(total, bg, sig)``
    return convention. Runs under ``torch.no_grad``, does not update
    weights, and restores the model to train mode afterwards.
    """
    model.eval()
    tot = bg_acc = sig_acc = 0.0
    for data, targets, mask in loader:
        data = data.to(device)
        targets = targets.float().to(device)
        mask = mask.to(device)
        preds = model(data)
        loss, bg_loss, sig_loss = _masked_mse_components(preds, targets, mask)
        tot += loss.item()
        bg_acc += bg_loss.item()
        sig_acc += sig_loss.item()

    model.train()
    n = max(1, len(loader))
    return tot / n, bg_acc / n, sig_acc / n


def eval_fn_td(
    loader,
    model,
    device,
    *,
    residual_channels: bool = False,
    residual_weight: float = 1.0,
) -> tuple:
    """Evaluate the two-detector time-domain separation model on a validation set.

    Mirrors the signature of :func:`train_fn_td` but runs under ``torch.no_grad``
    and does not update weights. Model is restored to train mode afterwards.

    Returns:
        (avg_total, avg_bg, avg_sig)           if residual_channels=False
        (avg_total, avg_bg, avg_sig, avg_res)  if residual_channels=True
    """
    loss_fn = nn.MSELoss()
    model.eval()
    tot = bg_acc = sig_acc = res_acc = 0.0
    n_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            preds = model(data)

            if residual_channels:
                pred_bg  = preds[:, 0:2]
                pred_sig = preds[:, 2:4]
                pred_res = preds[:, 4:6]
                tgt_bg   = targets[:, 0:2]
                tgt_sig  = targets[:, 2:4]
                res_tgt  = data - (tgt_bg + tgt_sig)
                bg_loss  = loss_fn(pred_bg,  tgt_bg)
                sig_loss = loss_fn(pred_sig, tgt_sig)
                res_loss = loss_fn(pred_res, res_tgt)
                loss = 0.5 * (bg_loss + sig_loss) + residual_weight * res_loss
                res_acc += res_loss.item() * data.size(0)
            else:
                pred_bg  = preds[:, 0:2]
                pred_sig = preds[:, 2:4]
                tgt_bg   = targets[:, 0:2]
                tgt_sig  = targets[:, 2:4]
                bg_loss  = loss_fn(pred_bg,  tgt_bg)
                sig_loss = loss_fn(pred_sig, tgt_sig)
                loss = 0.5 * (bg_loss + sig_loss)

            bs = data.size(0)
            tot     += loss.item()    * bs
            bg_acc  += bg_loss.item() * bs
            sig_acc += sig_loss.item() * bs
            n_samples += bs

    model.train()
    n = max(1, n_samples)
    if residual_channels:
        return tot / n, bg_acc / n, sig_acc / n, res_acc / n
    return tot / n, bg_acc / n, sig_acc / n
