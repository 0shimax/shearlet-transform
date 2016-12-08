import numpy as np

from meyer_shearlet import meyer_shearlet_spect, meyeraux
from _scales_shears_and_spectra import scales_shears_and_spectra
from _fft import fftshift, ifftshift, fftn, ifftn


def inverse_shearlet_transform_spect(ST, Psi=None, maxScale='max',
                                  shearletSpect=meyer_shearlet_spect,
                                  shearletArg=meyeraux):
    """Compute inverse shearlet transform.
    If the shearlet spectra, Psi, are not given they are computed using
    parameters guessed from the coefficients.
    Parameters
    ----------
    ST : array (3d)
        shearlet transform
    Psi : array (3d), optional
        3d spectrum of shearlets
    maxScale : {'min', 'max'}
        maximal or minimal finest scale
    shearletSpect : {meyerShearletSpect, meyerSmoothShearletSpect}
        shearlet spectrum to use
    shearletArg : function
        auxiliarry function for the shearlet
    Returns
    -------
    A : array (2d)
        reconstructed image
    """

    if Psi is None:
        # numOfScales
        # possible: 1, 4, 8, 16, 32,
        # -> -1 for lowpass
        # -> divide by for (1, 2, 4, 8,
        # -> +1 results in a 2^# number -> log returns #
        numOfScales = int(np.log2((ST.shape[-1] - 1)/4 + 1))

        # realCoefficients
        realCoefficients = True

        # realReal
        realReal = True

        # compute spectra
        Psi = scalesShearsAndSpectra((ST.shape[0], ST.shape[1]),
                                     numOfScales=numOfScales,
                                     realCoefficients=realCoefficients,
                                     realReal=realReal,
                                     shearletSpect=meyerShearletSpect,
                                     shearletArg=meyeraux)

    # inverse shearlet transform
    if False:
        # INCORRECT TO HAVE FFTSHIFT SINCE Psi ISNT SHIFTED!
        A = fftshift(fftn(ST, axes=(0, 1)), axes=(0, 1)) * Psi
        A = A.sum(axis=-1)
        A = ifftn(ifftshift(A))
    else:
        A = fftn(ST, axes=(0, 1)) * Psi
        A = A.sum(axis=-1)
        A = ifftn(A)

    if np.isrealobj(ST):
        A = A.real

    return A
