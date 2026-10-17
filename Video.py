import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import FreeParticle1D

GaussianWavePacket = FreeParticle1D.GaussianWavePacket
SplitStepFourierMethod = FreeParticle1D.SplitStepFourierMethod
V = FreeParticle1D.Potential
x, Lx, Nx, dt, Nt, k0 = FreeParticle1D.x, FreeParticle1D.Lx, FreeParticle1D.Nx, FreeParticle1D.dt, FreeParticle1D.Nt, FreeParticle1D.k0

dx = Lx/Nx

# Fourier space components
kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi

# Video settings: frames per second (fps) and output file
fps = 20
output_path = "Output/animation.mp4"

# Simulation steps shown per video frame (controls the playback speed)
steps_per_frame: int = 50
simulation_fps = steps_per_frame * fps  # Simulation steps per second of video

# Run the simulation and store the probability density once per video frame
psi0 = GaussianWavePacket(x0=0e-7, x = x, dx = dx, sigma_x=60e-9, kx0=k0)
psi = psi0.copy()

Rho_frames = []
for step in range(Nt):
    if step % steps_per_frame == 0:
        Rho_frames.append(np.abs(psi) ** 2)
    psi = SplitStepFourierMethod(psi=psi, dt=dt, x=x, dx=dx, kx=kx)
Rho_frames = np.array(Rho_frames)

nframe = len(Rho_frames)
Rho_0 = np.abs(psi0) ** 2

# Initialize figure
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(x[0] * 1e9, x[-1] * 1e9)
ax.set_xlabel(r'$x$ (nm)', fontsize=14)

# Density axis: limit from the maximum over ALL frames, since a well can focus the packet
ax.set_ylim(0, 1.1 * Rho_frames.max())
ax.set_ylabel(r'Probability Density (1/m)', fontsize=14)

# Potential on a second axis, in micro-eV
ax2 = ax.twinx()
ax2.plot(x * 1e9, V, color='red', linestyle='--', lw=2, label=r'$V(x)$')
ax2.set_ylabel(r'$V(x)$ ($\mu$eV)', fontsize=14)
ax2.set_ylim(1.1 * V.min(), -0.1 * V.min())

# Static reference: initial wave packet an Potential
# ax.plot(x * 1e9, Rho_0, color='red', linestyle='--', lw=2, label=r'$t = 0$')

# Animated line: current wave packet
(line,) = ax.plot([], [], color='hotpink', lw=2, label=r'$|\psi(x,t)|^2$')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, va='top')
ax.legend(loc='upper right', fontsize=12)


def innit():
    """Innitialize wavefunction for plotting"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def update(frame):
    """Update the plot data as a function of time."""
    t = frame / fps
    sim_frame = round(t * simulation_fps)  # Corresponding simulation time step

    line.set_data(x * 1e9, Rho_frames[frame])
    time_text.set_text(f'Time step: {sim_frame}  (t = {sim_frame * dt * 1e12:.2f} ps)')
    return line, time_text


# Give us a status bar
with tqdm() as progress:

    def my_progress(current_frame, total_frames):
        if progress.total != total_frames - 1:
            progress.reset(total=total_frames - 1)
        progress.update(current_frame - progress.n)

    ani = animation.FuncAnimation(fig=fig, func=update, init_func=innit,
                                  frames=nframe, interval=1000 / fps, blit=True)
    ani.save(output_path, fps=fps, progress_callback=my_progress)
    