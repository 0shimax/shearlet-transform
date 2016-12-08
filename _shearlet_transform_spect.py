import numpy as np

from meyer_shearlet import meyer_shearlet_spect, meyeraux

from _scales_shears_and_spectra import scales_shears_and_spectra
from _fft import fftshift, ifftshift, fftn, ifftn


def shearlet_transform_spect(A, Psi=None, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyer_shearlet_spect,
                           shearletArg=meyeraux, realReal=True):
    """Compute the forward shearlet transform.
    If the shearlet spectra, Psi, are not given they are computed using
    parameters guessed from the coefficients.
    Parameters
    ----------
    A : array
        image to transform (2d)
    Psi : array (3d), optional
        spectrum of shearlets (3d)
    realCoefficients : bool, optional
        force real-valued coefficients
    maxScale : {'min', 'max'}
        maximal or minimal finest scale
    shearletSpect : {meyerShearletSpect, meyerSmoothShearletSpect}
        shearlet spectrum to use
    shearletArg : function
        auxiliarry function for the shearlet
    realReal : bool, optional
        return coefficients with real dtype (truncate minimal imaginary
        component).
    Returns
    -------
    ST : array (2d)
        reconstructed image
    Psi : array (3d), optional
        spectrum of shearlets (3d)
    Notes
    -----
    example usage
    # compute shearlet transform of image A with default parameters
    ST, Psi = shearletTransformSpect(A)
    # compute shearlet transform of image A with precomputed shearlet spectrum
    ST, Psi = shearletTransformSpect(A, Psi)
    # compute sharlet transform of image A with specified number of scales
    ST, Psi = shearletTransformSpect(A, numOfScales=4)
    """
    # parse input
    A = np.asarray(A)
    if (A.ndim != 2) or np.any(np.asarray(A.shape) <= 1):
        raise ValueError("2D image required")

    # compute spectra
    if Psi is None:
        l = A.shape
        if numOfScales is None:
            numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
            if numOfScales < 1:
                raise ValueError('image to small!')
        Psi = scales_shears_and_spectra(l, numOfScales=numOfScales,
                                     realCoefficients=realCoefficients,
                                     shearletSpect=meyer_shearlet_spect,
                                     shearletArg=meyeraux)

    # shearlet transform
    if False:
        # INCORRECT TO HAVE FFTSHIFT SINCE Psi ISNT SHIFTED!
        uST = Psi * fftshift(fftn(A))[..., np.newaxis]
        ST = ifftn(ifftshift(uST, axes=(0, 1)), axes=(0, 1))
    else:
        uST = Psi * fftn(A)[..., np.newaxis]
        ST = ifftn(uST, axes=(0, 1))

    # due to round-off errors the imaginary part is not zero but very small
    # -> neglect it
    if realCoefficients and realReal and np.isrealobj(A):
        ST = ST.real

    return (ST, Psi)
