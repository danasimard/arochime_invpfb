from numpy.fft import rfft,irfft
import numpy as np


def sinc_window(ntap, lblock):
    """Sinc window function.
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    coeff_length = np.pi * ntap
    coeff_num_samples = ntap * lblock

    # Sampling locations of sinc function
    X = np.arange(-coeff_length / 2.0, coeff_length / 2.0,
                  coeff_length / coeff_num_samples)

    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so use X/pi
    return np.sinc(X / np.pi)


def sinc_hamming(ntap, lblock):
    """Hamming-sinc window function.
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """

    return sinc_window(ntap, lblock) * np.hamming(ntap * lblock)

def filter_kernel(nt):
    """Returns the Weiner Filtered PFB Kernel for the PFB inversion 
    Parameters
    ----------
    nt : int
        The number of time samples in the timestream for PFB inversion
    Returns
    ----------
    fw : np.ndarray
        The Weiner-Filtered PFB Kernel
        Dimensions: Time, -1
    """
    
    # S/N for use in the Wiener Filter
    # Assume 8 bits are set to have noise at 3 bits, 
    # so 1.5 bits for FT.
    # samples off by uniform distribution of [-0.5, 0.5] ->
    # equivalent to adding noise with std=0.2887
    prec = (1 << 3) ** 0.5
    sn = prec / 0.2887
    
    # Get deconvolution kernel
    nblock = 2048
    h = sinc_hamming(4, nblock).reshape(4, -1)
    h = np.pad(h,((0,nt-4),(0,0)),mode='constant',
               constant_values=(0.,))
    h = rfft(h, axis=0).conj()
    fw = h.conj() / (np.abs(h)**2 + (1/sn)**2)
    return fw 
 
def get_real_timestream(data):
    '''Returns the real pseudo-timestream 
    Parameters 
    ----------
    data : np.ndarray
        Nyquist-sample channelized data to process
        Dimensions: time, freq, pol
    Returns
    ----------
    data : np.ndarray
        The pseudo-timestream
        Dimensions: time, pol
    '''
    # Zero pad the nyquist frequency
    data = np.pad(data,((0,0),(1,0),(0,0)),
                  mode='constant',constant_values=(0.,))
    # Inverse FT to get real pseudo-timestream
    data = irfft(data, axis=1).reshape(-1,data.shape[-1])
    
    return data
    
def deconvolve_timestream(data,nt):
    '''Deconvolves a timestream by the Weiner-filtered inverse 
        PFB function
    Parameters
    ----------
    data : np.ndarray
        The pseudo-timestream
        Dimensions: time, pol
    nt : int
        The number of time bins in the original, 
        1024 channels data
    Returns
    ----------
    data : np.ndarray
        The timestream after removing the PFB
        Dimensions: time, pol
    '''
    # Deconvolve and get deconvolved timestream
    npol = data.shape[-1]
    rd = irfft(rfft(data.reshape(nt,-1,npol),axis=0) * 
               filter_kernel(nt)[..., np.newaxis],
                      axis=0).reshape(-1, npol)

    return rd

def pfb(timestream, nfreq, ntap=4, window=sinc_hamming):
    """Perform the CHIME PFB on a timestream.
    Parameters
    ----------
    timestream : np.ndarray
        Timestream to process
    nfreq : int
        Number of frequencies we want out (probably should be odd
        number because of Nyquist)
    ntaps : int
        Number of taps.
    Returns
    -------
    pfb : np.ndarray[:, nfreq]
        Array of PFB frequencies.
    """

    # Number of samples in a sub block
    lblock = 2 * (nfreq - 1)

    # Number of blocks
    nblock = timestream.size // lblock - (ntap - 1)

    # Initialise array for spectrum
    spec = np.zeros((nblock, nfreq), dtype=np.complex128)

    # Window function
    w = window(ntap, lblock)

    # Iterate over blocks and perform the PFB
    for bi in range(nblock):
        # Cut out the correct timestream section
        ts_sec = timestream[(bi*lblock):((bi+ntap)*lblock)].copy()

        # Perform a real FFT (with applied window function)
        ft = np.fft.rfft(ts_sec * w)

        # Choose every n-th frequency
        spec[bi] = ft[::ntap]

    return spec