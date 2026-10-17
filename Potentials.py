import numpy as np

eV = 1.602176634e-19  # Joule

def PoschlTellerPotential(x, V0: float = 5e-6 * eV, width: float = 100e-9):
    """
    Return a Poschl-Teller potential well (J), centered in the domain

    Parameters
    ----------
    V0 : float
        Depth of the well (Joule)
    width : float
        Characteristic width of the well (meters)
    """
    x_mid = (x.max() + x.min()) / 2.
    V = -V0 / np.cosh((x - x_mid) / width) ** 2
    return V/eV

def BoxPotential(x, L: float = 300e-9, xc: float = 0., V0: float = 1e-3 * eV):
    """
    Box potential (Joule): 0 inside xc - L/2 < x < xc + L/2, hard wall V0 outside

    Parameters
    ----------
    L : float
        Length of the box (meters)
    xc : float
        Center of the box (meters)
    V0 : float
        Wall height (Joule), must be >> kinetic energy of the packet
    """
    V_x = np.full_like(x, V0)
    V_x[np.abs(x - xc) < L / 2] = 0.
    return V_x

def BarrierPotential(x, V0: float = 300e-6 * eV, width: float = 30e-9, xc: float = 0.):
    """
    Rectangular potential barrier (Joule): V0 for |x - xc| < width/2, 0 elsewhere

    Parameters
    ----------
    V0 : float
        Barrier height (Joule)
    width : float
        Barrier width (meters)
    xc : float
        Center of the barrier (meters)
    """
    V_x = np.zeros_like(x)
    V_x[np.abs(x - xc) < width / 2] = V0
    return V_x
