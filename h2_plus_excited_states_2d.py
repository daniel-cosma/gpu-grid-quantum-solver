

import cupy as cp
from PIL import Image

import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as cspla
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cupy")
N = 400  # grid N*N
dx = 0.5

# 1.  2D laplacian (5 points)
main_diag = -4.0 * cp.ones(N*N)
offsets = cp.array([-N, -1, 0, 1, N])
L = csp.diags([1, 1, main_diag, 1, 1], offsets, shape=(N*N, N*N), format='csr')

# 2.  V(x,y) (on gpu)
x = cp.linspace(-N/2, N/2, N)
X, Y = cp.meshgrid(x, x)
# 2 nuclei
d = 0.0  # center distance
Z = 0.5  #charge of the nuclei
eps = 0.02  # coulombian smoothing (to avoid 0 division)

# get distance from the nuclei
r1 = cp.sqrt((X - d) ** 2 + Y**2)
r2 = cp.sqrt((X + d) ** 2 + Y**2)


V = -Z / (r1 + eps) - Z / (r2 + eps)
V_diag = csp.diags(V.flatten())

# Hamiltonian H = -0.5 * L + V
H = -0.5 * L / (dx**2) + V_diag


# 3.  Lanczos solver on GPU (CuPy / cuSOLVER)
i=20 # ---------edit this value to get the desired ammount of eigenvalues-------


energies, wavefunctions = cspla.eigsh(H, k=i, which='SA', tol=1e-12, maxiter=8000)
for x in range (i):
    
    orbita = wavefunctions[:,x].reshape((N, N)).get() # .get() -> CuPy array to NumPy
    plt.imsave(f'My_H_atom_{x}.png', orbita*orbita, cmap='magma')

plt.imshow(orbita**2, cmap='magma')

plt.axis('off')
plt.show()
