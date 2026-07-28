"""Monte Carlo Dropout utilities for uncertainty estimation.

At test time, keeping Dropout active and running N stochastic forward passes
gives N samples from an approximate posterior over model outputs. Works with
both UNET2D (Dropout2d, STFT domain) and UNET1D (Dropout1d, time domain).

**For STFT-domain models (UNET2D), work in time domain, not STFT domain.**
The model outputs magnitude+phase spectrograms. Phase is a circular quantity,
so naive averaging across passes in the STFT domain is incorrect (e.g. phases
of +170 deg and -170 deg average to 0 deg, not +-180 deg). The right approach is to
apply iSTFT to each sample *before* aggregating, so the posterior is over
time-domain reconstructions — which is also the output space you care about
physically. UNET1D has no such concern since it already outputs time-domain
samples directly.

Usage (STFT / UNET2D)::

    from deepextractor.utils import enable_mc_dropout, mc_predict
    from deepextractor.utils.stft import apply_istft

    model.eval()              # freeze BatchNorm running stats
    enable_mc_dropout(model)  # re-enable Dropout2d only

    import functools
    istft_fn = functools.partial(
        apply_istft, n_fft=512, hop_length=32, win_length=64,
        window=torch.hann_window(64),
    )

    # Full posterior in time domain — shape (n_passes, N, L)
    samples = mc_predict(model, x, n_passes=100, postprocess_fn=istft_fn)

    mean   = samples.mean(dim=0)            # shape (N, L)
    median = samples.median(dim=0).values
    std    = samples.std(dim=0)             # per-sample uncertainty vs time
    lo, hi = samples.quantile(torch.tensor([0.05, 0.95]), dim=0)

    # Or get a summary directly
    mean = mc_predict(model, x, n_passes=100, postprocess_fn=istft_fn, reduction="mean")

Usage (time domain / UNET1D) — no postprocess_fn needed, model output is
already time domain::

    model.eval()
    enable_mc_dropout(model)
    samples = mc_predict(model, x, n_passes=100)   # (n_passes, N, C, L)

"""

import torch
import torch.nn as nn
from typing import Callable, Optional


def enable_mc_dropout(model: nn.Module) -> None:
    """Set all Dropout1d/Dropout2d modules to train mode while leaving everything else eval.

    Call this after ``model.eval()`` to activate Monte Carlo Dropout at
    inference time. BatchNorm layers stay in eval mode so they use their
    running statistics rather than per-batch statistics. Works for both
    UNET2D (Dropout2d) and UNET1D (Dropout1d).
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout1d, nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 100,
    postprocess_fn: Optional[Callable] = None,
    reduction: str = "none",
) -> torch.Tensor:
    """Draw *n_passes* stochastic samples from the approximate posterior.

    Each forward pass uses a different dropout mask. An optional
    ``postprocess_fn`` is applied to each raw model output before stacking —
    use this to convert from STFT domain to time domain via iSTFT (recommended
    for UNET2D, see module docstring for why averaging in STFT domain is
    incorrect). Not needed for UNET1D, whose output is already time domain.

    Parameters
    ----------
    model          : nn.Module with Dropout1d/Dropout2d layers already in train
                     mode (call ``enable_mc_dropout`` first).
    x              : torch.Tensor — input batch, any rank (N, C, L) for UNET1D
                     or (N, C, H, W) for UNET2D; the batch dimension (dim 0)
                     is what gets repeated across passes.
    n_passes       : int — number of stochastic forward passes (50-100 typical).
    postprocess_fn : optional callable applied to each model output before
                     stacking, e.g. ``functools.partial(apply_istft, ...)``
                     to convert STFT outputs to time-domain signals.
    reduction      : str — how to summarise the stacked samples:
                       ``"none"``   — return all samples (recommended for
                                      forming a full posterior)
                       ``"mean"``   — sample-wise mean
                       ``"median"`` — sample-wise median

    Returns
    -------
    torch.Tensor
        If ``reduction="none"``: shape ``(n_passes, N, ...)`` where ``...``
        is the shape of a single (optionally post-processed) output.
        If ``reduction="mean"`` or ``"median"``: shape ``(N, ...)``.

    Examples
    --------
    >>> samples = mc_predict(model, x, n_passes=100, postprocess_fn=istft_fn)
    >>> mean   = samples.mean(dim=0)
    >>> median = samples.median(dim=0).values
    >>> std    = samples.std(dim=0)    # uncertainty map over time
    >>> lo, hi = samples.quantile(torch.tensor([0.05, 0.95]), dim=0)
    """
    # Batch all passes into a single forward call: repeat the input n_passes times
    # along the batch dimension so each copy gets an independent dropout mask.
    # This is much faster than n_passes sequential calls — one kernel launch
    # instead of n_passes, and the GPU/CPU can parallelise across the batch.
    # BatchNorm stays in eval mode (uses running stats) so batch size doesn't
    # affect the normalisation; Dropout in train mode samples independently
    # per batch element, giving us n_passes distinct stochastic reconstructions.
    # repeat_dims matches x's actual rank (works for UNET1D's (N,C,L) and
    # UNET2D's (N,C,H,W) alike), unlike a hardcoded 4-tuple.
    N = x.shape[0]
    repeat_dims = (n_passes,) + (1,) * (x.dim() - 1)
    x_rep = x.repeat(*repeat_dims)                  # (P*N, ...)
    raw_flat = model(x_rep)                          # (P*N, ...)
    # Restore (P, N, ...): repeat tiles [s0,s1,...,s0,s1,...] so we permute
    raw = raw_flat.view(n_passes, N, *raw_flat.shape[1:])

    if postprocess_fn is not None:
        P = raw.shape[0]
        merged = raw.view(P * N, *raw.shape[2:])           # (P*N, ...)
        processed = postprocess_fn(merged)                  # (P*N, ...)
        samples = processed.view(P, N, *processed.shape[1:])  # (P, N, ...)
    else:
        samples = raw

    if reduction == "mean":
        return samples.mean(dim=0)
    if reduction == "median":
        return samples.median(dim=0).values
    if reduction == "none":
        return samples
    raise ValueError(f"reduction must be 'none', 'mean', or 'median' — got {reduction!r}")
