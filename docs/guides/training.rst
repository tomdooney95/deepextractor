Training
========

Glitch reconstruction dataset
------------------------------

DeepExtractor reconstructions of seven LIGO O3 glitch classes (35,000 samples) are
available for download from HuggingFace:
`tomdooney/deepextractor-glitch-reconstructions <https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions>`_

These reconstructions are suitable for training generative models such as GlitchGAN
and for any downstream task requiring high-quality time-domain glitch waveforms.

.. code-block:: python

   from deepextractor.data import download_glitch_data

   paths = download_glitch_data("data/glitches/")
   # paths["samples"]     → (35000, 8192) whitened waveforms
   # paths["labels"]      → (35000, 7)   one-hot class labels
   # paths["label_order"] → ['Blip', 'Fast_Scattering', 'Koi_Fish',
   #                          'Low_Frequency_Burst', 'Scattered_Light', 'Tomte', 'Whistle']

To also download the first-order time-derivative array (required for cDVGAN training):

.. code-block:: python

   paths = download_glitch_data("data/glitches/", include_derivatives=True)
   # paths["derivatives"] → (35000, 8191)

Data preparation
----------------

1. Generate time-domain data:

.. code-block:: bash

   deepextractor-generate --output-dir data/ --num-train 250000 --num-val 25000

   # Or with bilby noise:
   deepextractor-generate --output-dir data/ --num-train 250000 --bilby-noise

2. Convert to spectrograms (for 2D models):

.. code-block:: bash

   deepextractor-specgen \
       --input-dir data/pycbc_noise/time_domain/ \
       --output-dir data/pycbc_noise/spectrogram_domain/

Expected directory layout after data generation::

   data/
   └── pycbc_noise/
       ├── time_domain/
       │   ├── glitch_train_scaled_pycbc.npy
       │   ├── background_train_scaled_pycbc.npy
       │   ├── glitch_val_scaled_pycbc.npy
       │   └── background_val_scaled_pycbc.npy
       └── spectrogram_domain/
           ├── glitch_train_scaled_mag_phase.npy
           ├── background_train_scaled_mag_phase.npy
           ├── glitch_val_scaled_mag_phase.npy
           └── background_val_scaled_mag_phase.npy

Training a model
----------------

.. code-block:: bash

   deepextractor-train \
       --model DeepExtractor_257 \
       --data-dir data/pycbc_noise/spectrogram_domain/ \
       --checkpoint-dir checkpoints/ \
       --batch-size 32 \
       --epochs 150

Available models
----------------

.. list-table::
   :header-rows: 1

   * - Model name
     - Architecture
     - Domain
   * - ``DeepExtractor_257``
     - UNET2D (257×257 spectrograms)
     - Spectrogram
   * - ``DeepExtractor_129``
     - UNET2D (129×129 spectrograms)
     - Spectrogram
   * - ``UNET1D``
     - 1D U-Net
     - Time-domain
   * - ``DnCNN1D``
     - 1D DnCNN
     - Time-domain
   * - ``Autoencoder1D``
     - 1D Autoencoder
     - Time-domain

Hyperparameter options
----------------------

Run ``deepextractor-train --help`` for the full list of arguments.
