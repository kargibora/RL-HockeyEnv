import torch
from typing import Union, Iterable, Optional
from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum

def powerlaw_psd_gaussian(
        exponent: float, 
        size: Union[int, Iterable[int]], 
        fmin: float = 0.0, 
        random_state: Optional[Union[int, Generator, RandomState]] = None
    ):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. 
        
        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState, 
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples) # type: ignore # mypy 1.5.1 has problems here 
    
    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples) # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    # Build scaling factors for all frequencies
    s_scale = f    
    ix   = npsum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    
    # prepare random number generator
    normal_dist = _get_normal_distribution(random_state)

    # Generate scaled random power + phase
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)    # Fix magnitude
    
    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)    # Fix magnitude
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    
    return y

def _get_normal_distribution(random_state: Optional[Union[int, Generator, RandomState]]):
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError(
            "random_state must be one of integer, numpy.random.Generator, "
            "numpy.random.Randomstate"
        )
    return normal_dist

def sample_colored_noise(shape: tuple, beta: float) -> torch.Tensor:
    noise = powerlaw_psd_gaussian(beta, shape)
    return torch.from_numpy(noise)

def sample_gaussian_noise(shape: tuple) -> torch.Tensor:
    return torch.normal(torch.zeros(shape), 1.0)

def sample_golden_noise(shape: tuple) -> torch.Tensor:
    return sample_colored_noise(shape, beta=(1 + sqrt(5)) / 2)

def sample_pink_noise(shape: tuple) -> torch.Tensor:
    return sample_colored_noise(shape, beta=1.0)

def sample_brown_noise(shape: tuple) -> torch.Tensor:
    return sample_colored_noise(shape, beta=2.0)

