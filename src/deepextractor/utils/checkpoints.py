import os
import logging

import torch

logger = logging.getLogger(__name__)

# Hugging Face Hub repo that hosts pretrained weights.
HF_REPO_ID = "tomdooney/deepextractor"

# Named checkpoint constants — use these instead of bare filename strings.
CHECKPOINT_BILBY = "checkpoint_best_bilby_noise_base.pth.tar"  # trained on simulated bilby noise
CHECKPOINT_REAL  = "checkpoint_best_real_noise_base.pth.tar"   # fine-tuned on real O3 LIGO data


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Save model and optimizer state as a checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """Load model state from a checkpoint."""
    print("=> Loading checkpoint")
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except KeyError:
        print("=> Failed to load checkpoint: state_dict not found")
        raise


def load_optimizer(checkpoint, optimizer):
    """Load optimizer state from a checkpoint."""
    print("=> Loading optimizer")
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except KeyError:
        print("=> Failed to load checkpoint: optimizer state not found")
        raise


def _resolve_checkpoint(model_name, checkpoint_dir, filename=CHECKPOINT_BILBY):
    """
    Return the local path to a checkpoint file.

    Checks ``checkpoint_dir`` first; falls back to downloading from Hugging Face Hub.
    """
    if checkpoint_dir is not None:
        local_path = os.path.join(checkpoint_dir, model_name, filename)
        if os.path.isfile(local_path):
            return local_path

    # Fall back to Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download pretrained weights. "
            "Install it with: pip install huggingface_hub"
        ) from e

    hf_path = f"{model_name}/{filename}"
    logger.info(f"Downloading {hf_path} from {HF_REPO_ID} ...")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=hf_path)


def load_torch_model(model_name, model_dict, checkpoint_dir=None, device="cpu",
                     checkpoint_filename=CHECKPOINT_BILBY):
    """
    Load a pretrained PyTorch model.

    Weights are resolved in this order:

    1. ``<checkpoint_dir>/<model_name>/<checkpoint_filename>`` (local)
    2. Hugging Face Hub (``tomdooney/deepextractor``) — downloaded and cached automatically.

    Use the module-level constants ``CHECKPOINT_BILBY`` and ``CHECKPOINT_REAL`` for
    ``checkpoint_filename`` to select the correct variant.

    Args:
        model_name (str): Name of the model (must be a key in ``model_dict``).
        model_dict (dict): Mapping of model names to instantiated model objects.
        checkpoint_dir (str | None): Local directory to search first. Pass ``None``
            to skip local lookup and always use Hugging Face Hub.
        device (str | torch.device): Device to load the model onto.
        checkpoint_filename (str): Checkpoint file name inside the model subdirectory.

    Returns:
        torch.nn.Module: The model with loaded weights in eval mode, or ``None`` on failure.
    """
    try:
        model = model_dict[model_name].to(device)
        checkpoint_path = _resolve_checkpoint(model_name, checkpoint_dir, checkpoint_filename)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval()
        logger.info(f"Successfully loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading model checkpoint for {model_name}: {e}")
        return None

    return model
