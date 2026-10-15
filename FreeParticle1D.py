import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34  # Reduced Planck constant (Joule second)
m_e = 9.10938356e-31  # Mass of electron (kg)
wavelength = 5e-9  # de Broglie wavelength (meters)
k0 = 2 * np.pi / wavelength  # Wave number
k0 = 0

# Grid parameters
Lx: float = 1e-6  # Simulation domain size (meters)
Nx: int = 1024  # Number of grid points
x = np.linspace(-Lx / 2, Lx / 2, Nx)
dx = Lx / Nx

# Time parameters
dt: float  = 3e-15  # Time step (seconds)
Nt: int = 10001  # Number of time steps

def GaussianWavePacket(x0, sigma_x, x, dx, kx0:float = 0, npw=np):
    """
    Return a Gaussian wave packet at t = 0

    Parameters
    ----------
    x0 : float
        Initial position of the packet (meters)
    sigma_x : float
        Width of the Gaussian packet (meters)
    Nx : int
        Number of grid points
    Lx : float
        Simulation domain size (meters)
    kx0 : float
        Initial wave vector
    npw : module
        NumPy wrapper to be used for calculations

    Returns
    -------
    psi0 : float
        Value of the normalised wave function describing a Gaussian wave packet at t = 0
    """
    # Initial Gaussian wave packet
    psi0 = npw.exp(-(x - x0) ** 2 / (2 * sigma_x ** 2)) * npw.exp(1j * kx0 * x)
    psi0 = psi0.astype(npw.complex128)
    # Normalize the initial wave packet
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)
    return psi0

psi0 = GaussianWavePacket(x0 = 0e-7, sigma_x = 60e-9, x = x, dx = dx, kx0 = k0)

def SplitStepFourierMethod(psi, dt, x, dx, kx, npw=np):
    """
    Calculate the propagation of a wave function using the SSFM methode.

    Parameters
    ----------
    psi : float
        Initial wave function, before the time step
    dt : float
        Time step (seconds)
    x : float
        (meters)
    dx : float
        Distance step (meters)
    kx : float
        fourier space components ()
    npw : module
        NumPy wrapper to be used for calculations

    Returns
    -------
    psi : float
        Value of the propagated wave function, i.e. psi(t + dt)
    """

    # (a) 1/2 Evolution for the potential energy in real space
    # we can skip this step for now since V = 0
    
    # (b) Forward transform
    psi_k = npw.fft.fft(psi)
    
    # (c) Full evolution for the kinetic energy in Fourier space
    psi_k *= npw.exp(-1j * hbar * kx**2 * dt / (2 * m_e))
    
    # (d) Inverse Fourier transform
    psi = npw.fft.ifft(psi_k)
    
    # (e) 1/2 Evolution for the potential energy in real space
    # we can skip this step for now since V = 0
    return psi

# If the user runs the file itself, make a brute visualisation so the video can be checked
if __name__ == '__main__':
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    psi0 = GaussianWavePacket(x0 = 0e-7, sigma_x = 60e-9, kx0 = k0)
    psi = psi0.copy()

    for t in range(Nt):
        psi = SplitStepFourierMethod(psi=psi, dt = dt, Lx = Lx, Nx = Nx)
        # Visualization of wave packet evolution
        if t % 500 == 0:
            plt.figure(figsize=(8, 6))
            Rho = np.abs(psi) ** 2
            plt.plot(x * 1e9, Rho, color='hotpink', lw=2)
            Rho_0 = np.abs(psi0) ** 2; plt.plot(x * 1e9, Rho_0, color='red', linestyle = '--', lw=2)
            plt.title(f'Time step: {t}')
            plt.xlabel(r'$x$ (nm)', fontsize=14)
            plt.ylabel(r'Probability Density', fontsize=14)
            plt.pause(1)  # Pause for half a second
            plt.close()

    plt.show()


