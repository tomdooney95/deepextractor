"""Monte Carlo Dropout utilities for uncertainty estimation.

At test time, keeping Dropout active and running N stochastic forward passes
gives N samples from an approximate posterior over model outputs.

**Important — work in time domain, not STFT domain.**
The model outputs magnitude+phase spectrograms.  Phase is a circular quantity,
so naive averaging across passes in the STFT domain is incorrect (e.g. phases
of +170° and -170° average to 0°, not ±180°).  The right approach is to apply
iSTFT to each sample *before* aggregating, so the posterior is over time-domain
reconstructions — which is also the output space you care about physically.

Usage::

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

"""

import torch
import torch.nn as nn
from typing import Callable, Optional


def enable_mc_dropout(model: nn.Module) -> None:
    """Set all Dropout2d modules to train mode while leaving everything else eval.

    Call this after ``model.eval()`` to activate Monte Carlo Dropout at
    inference time.  BatchNorm layers stay in eval mode so they use their
    running statistics rather than per-batch statistics.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
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

    Each forward pass uses a different dropout mask.  An optional
    ``postprocess_fn`` is applied to each raw model output before stacking —
    use this to convert from STFT domain to time domain via iSTFT (recommended,
    see module docstring for why averaging in STFT domain is incorrect).

    Parameters
    ----------
    model          : nn.Module with Dropout2d layers already in train mode
                     (call ``enable_mc_dropout`` first).
    x              : torch.Tensor, shape (N, C, H, W) — input batch.
    n_passes       : int — number of stochastic forward passes (50–100 typical).
    postprocess_fn : optional callable applied to each model output before
                     stacking, e.g. ``functools.partial(apply_istft, ...)``
                     to convert STFT outputs to time-domain signals.
    reduction      : str — how to summarise the stacked samples:
                       ``"none"``   — return all samples (recommended for
                                      forming a full posterior)
                       ``"mean"``   — pixel/sample-wise mean
                       ``"median"`` — pixel/sample-wise median

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
    # Collect all raw model outputs first, then post-process in one batched call.
    # For iSTFT this is faster than applying it per-pass in the loop because
    # the full (n_passes * N) batch is processed in a single kernel launch.
    raw = torch.stack([model(x) for _ in range(n_passes)], dim=0)  # (P, N, ...)

    if postprocess_fn is not None:
        P, N = raw.shape[0], raw.shape[1]
        merged = raw.view(P * N, *raw.shape[2:])           # (P*N, 2, F, T)
        processed = postprocess_fn(merged)                  # (P*N, L)
        samples = processed.view(P, N, *processed.shape[1:])  # (P, N, L)
    else:
        samples = raw

    if reduction == "mean":
        return samples.mean(dim=0)
    if reduction == "median":
        return samples.median(dim=0).values
    if reduction == "none":
        return samples
    raise ValueError(f"reduction must be 'none', 'mean', or 'median' — got {reduction!r}")
