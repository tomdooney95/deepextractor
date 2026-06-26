import numpy as np
import matplotlib.pyplot as plt

"""
Script to extract various kinds glitches from the glitch bank (glitch_GAN_samples_scaled_balanced.npy)

The bank is created by extracting the glitch time series from the LIGO-Virgo data using deepextractor, i.e., they are real glitches as seen by LIGO-Virgo.


# order = ['Blip', 'Fast_Scattering', 'Koi_Fish', 'Low_Frequency_Burst',
#           'Scattered_Light', 'Tomte', 'Whistle']

For example:

scattered_light = samples[labels[:, 4] == 1]  # 5000 samples, shape (5000, 8192)
"""


def plot_glitches(samples, n=10, filename='output/scattered_light_glitches.pdf'):
    """
    Function for plotting glitches. 
    """
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    t = np.arange(samples.shape[1]) / 4096
    for i, ax in enumerate(axes):
        ax.plot(t, samples[i])
        ax.set_ylabel(f'#{i}', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Scattered Light Glitches')
    fig.tight_layout()
    fig.savefig(filename)
    print(f'Saved {filename}')

def main():
    #location: `CIT:/home/tom.dooney/cDVGAN_for_DeepExtractor/data/glitch_GAN_samples_scaled_balanced.npy`
    # TODO: Move the files to the hugging face release
    samples = np.load('./glitch_GAN_samples_scaled_balanced.npy')
    labels = np.load('./glitch_GAN_labels_balanced.npy')
    scattered_light = samples[labels[:, 4] == 1]  # 5000 samples, shape (5000, 8192)

    plot_glitches(scattered_light[:10])

    t = np.arange(scattered_light.shape[1]) / 4096

    # Some indices to save the *npy data to *dat file
    for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 833]:
        np.savetxt(f'output/scattered_light_glitch_{idx}.dat',
                   np.column_stack([t, scattered_light[idx]]),
                   header='time strain')



    
if __name__ == "__main__":
    main()