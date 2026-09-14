import numpy as np
import bilby

def compute_scale(frequency_mask, sampling_frequency):
    '''
    compute the scale factor that is required to do fft of the whiten data
    this is required only for bilby implementation
    '''
    x = np.sum(frequency_mask) / len(frequency_mask)
    y = np.sqrt(np.sum(frequency_mask))

    factor = sampling_frequency / (y/x)

    return factor


def nfft(time_domain_strain, sampling_frequency, minimum_frequency, duration):
    '''
    Wrapper for the bilby function to convert time domain to frequency domain

    Parameters:
    -----------
    time_domain_strain : array_like
        The time-domain strain data to be converted to the frequency domain.
        
    sampling_frequency : float
        The sampling frequency of the time-domain signal in Hz.
        
    minimum_frequency : float
        The minimum frequency threshold. Frequencies below this value will be set to zero 
        in the output frequency domain data.
        
    duration : float
        The duration of the time-domain signal in seconds.

    Returns:
    --------
    frequency_domain_strain : array_like
        The frequency-domain representation of the input time-domain strain data,
        with frequencies below `minimum_frequency` set to zero.
        
    frequency_array : array_like
        The array of frequency values corresponding to `frequency_domain_strain`.

    '''
    
    full_frequency = np.arange(0, (0.5*sampling_frequency) + 1/duration, 1/duration)
    frequency_mask = ((full_frequency >= minimum_frequency) & (full_frequency <= 0.5*sampling_frequency))

    window = compute_scale(frequency_mask, sampling_frequency)


    frequency_domain_strain, frequency_array = bilby.core.utils.nfft(time_domain_strain*window, sampling_frequency)

    ### Since we are not interested in the data below the minimum_frequency...
    frequency_domain_strain[np.where(frequency_array<minimum_frequency)] = 0.

    return frequency_domain_strain, frequency_array



def white_time_domain_strain_to_colored_frequency_domain_strain(white_time_domain_strain, asd, sampling_frequency = 4096, minimum_frequency = 20, duration = 4):
    """
    Utility function to colour whitened time domain strain
    """

    white_frequency_domain_strain, frequency_array = nfft(white_time_domain_strain, sampling_frequency, minimum_frequency, duration)

    interp_asd = np.interp(frequency_array, asd[0], asd[1])
    interp_asd[np.isinf(interp_asd)] = 0.


    coloured_frequency_domain_strain = white_frequency_domain_strain*interp_asd*np.sqrt(duration/4)

    return coloured_frequency_domain_strain