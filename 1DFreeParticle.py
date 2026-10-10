import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34  # Reduced Planck constant (Joule second)
m_e = 9.10938356e-31  # Mass of electron (kg)
wavelength = 5e-9  # de Broglie wavelength (meters)
k0 = 2 * np.pi / wavelength  # Wave number
k0 = 0

# Grid parameters
Lx = 1e-6  # Simulation domain size (meters)
Nx = 1024  # Number of grid points
dx = Lx / Nx  # Grid spacing
x = np.linspace(-Lx / 2, Lx / 2, Nx)

# Time parameters
dt = 3e-15  # Time step (seconds)
Nt = 10001  # Number of time steps

# Wave packet parameters
sigma_x = 60e-9  # Width of the Gaussian packet (meters)
x0 = 0e-7  # Initial position of the packet (meters)
kx0 = k0  # Initial wave vector

# Initial Gaussian wave packet
psi0 = np.exp(-(x - x0) ** 2 / (2 * sigma_x ** 2)) * np.exp(1j * kx0 * x)
psi0 = psi0.astype(np.complex128)

# Normalize the initial wave packet
psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)

# Fourier space components
kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
K2 = kx ** 2

# Split-step Fourier method
psi = psi0.copy()

for t in range(Nt):
    # (a) 1/2 Evolution for the potential energy in real space
    # we can skip this step for now since V = 0
    
    # (b) Forward transform
    psi_k = np.fft.fft(psi)
    
    # (c) Full evolution for the kinetic energy in Fourier space
    psi_k *= np.exp(-1j * hbar * K2 * dt / (2 * m_e))
    
    # (d) Inverse Fourier transform
    psi = np.fft.ifft(psi_k)
    
    # (e) 1/2 Evolution for the potential energy in real space
    # we can skip this step for now since V = 0
    
    
    # Visualization of wave packet evolution
    if t % 500 == 0:
        plt.figure(figsize=(8, 6))
        Rho = np.abs(psi) ** 2
        plt.plot(x * 1e9, Rho / np.max(Rho), color='hotpink', lw=2)
        plt.title(f'Time step: {t}')
        plt.xlabel(r'$x$ (nm)', fontsize=14)
        plt.ylabel(r'Probability Density', fontsize=14)
        plt.pause(1)  # Pause for 1 second
        plt.close()

plt.show()
