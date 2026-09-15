import warnings
import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as cspla
import matplotlib.pyplot as plt
from pyevtk.hl import imageToVTK

warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

# 1. 3D physical grid parameters
N = 256 #  N^3 Grid on a 3060 i suggest to use N<180
dx = 0.1
num_nodes = N * N * N

# 2.  Laplacian 3D  (7 points)
main_diag = -6.0 * cp.ones(num_nodes)
offsets = cp.array([-N * N, -N, -1, 0, 1, N, N * N])
ones_vec = cp.ones(num_nodes)

L = csp.diags(
    [ones_vec, ones_vec, ones_vec, main_diag, ones_vec, ones_vec, ones_vec],
    offsets,
    shape=(num_nodes, num_nodes),
    format="csr",
)

# 3. Potential V(x,y,z) 3D
x = cp.linspace(-N / 2, N / 2, N) * dx
X, Y, Z = cp.meshgrid(x, x, x, indexing="ij")

d_val = 1.0  #nuclear separation
Z_charge = 1.0  # nuclear charge
eps = 0.02  #  Coulombian smooting

r1 = cp.sqrt((Z - d_val) ** 2 + Y**2 + X**2)
r2 = cp.sqrt((Z + d_val) ** 2 + Y**2 + X**2)

V = -Z_charge / (r1 + eps) - Z_charge / (r2 + eps)
V_diag = csp.diags(V.flatten())

# Hamiltonian H = -0.5 * L / dx^2 + V
H = -0.5 * (L / (dx**2)) + V_diag

# 4.  VTK config and Solver ( Lanczos problably)
origin = (float(x[0]), float(x[0]), float(x[0]))
spacing = (dx, dx, dx)
num_eigenvalues = 1

energies, wavefunctions = cspla.eigsh(
    H, k=num_eigenvalues, which="SA", tol=1e-12, maxiter=10000
)

# 5. Export 
for state_idx in range( num_eigenvalues):
    #probability density |psi|^2 on the GPU
    psi_sq_gpu = cp.abs(wavefunctions[:, state_idx]) ** 2
    psi_sq_norm = psi_sq_gpu / cp.max(psi_sq_gpu)
    orbita_3d = psi_sq_norm.reshape((N, N, N)).get()  # -> NumPy
  

    # Export to ParaView (.vti)
    imageToVTK(
        f"./H2_orbital_{state_idx}",
        origin=origin,
        spacing=spacing,
        cellData=None,
        pointData={"psi_sq": orbita_3d},
    )
    print(f"[+] eigenstate {state_idx + 1}/{num_eigenvalues} done!", flush=True)
# Visualizzazione di una sezione 2D centrale (fetta Z) dell'ultimo stato
plt.figure(figsize=(6, 6))
plt.imshow(orbita_3d[:, :, N // 2], cmap="magma", origin="lower")
plt.title(f"state {num_eigenvalues-1} - section Z=0")
plt.axis("off")
plt.show()
