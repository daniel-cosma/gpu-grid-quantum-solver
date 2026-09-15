## gpu-grid-quantum-solver
2D/3D CuPy GPU-accelerated quantum solver for atomic/molecular orbitals
## Usage & Visualization

Running `h2_plus_excited_states_3d.py` generates 3D scalar field dataset files (`.vti`) directly compatible with **ParaView** (tested on version `5.10.1`).

To replicate the rendering results shown in the images below:
1. Open the generated `.vti` file in ParaView.
2. Apply the **Contour** filter.
3. Set **10 linearly spaced isosurfaces/isolines**.
4. Adjust the display **Opacity** to `0.2`.

---

## Requirements & Dependencies

This solver leverages GPU acceleration via CUDA. 

- **Hardware:** NVIDIA GPU with CUDA support (Compute Capability 3.5+).
- **Python Packages:** Dependencies are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
<img width="1670" height="1148" alt="Render 0006" src="https://github.com/user-attachments/assets/e7d9d4eb-e554-4eda-8f2a-cb0b4a7324f6" />
<img width="1670" height="1148" alt="Render 0005" src="https://github.com/user-attachments/assets/e45619c9-ee7e-441a-bf8b-8f2861f31a9b" />
<img width="1670" height="1148" alt="Render 0004" src="https://github.com/user-attachments/assets/79ae0786-757c-45dc-82b4-6fa37ada6881" />
<img width="1670" height="1148" alt="Render 0003" src="https://github.com/user-attachments/assets/67a88f7e-faba-4586-a669-a88b465ccf79" />
<img width="1670" height="1148" alt="Render 0002" src="https://github.com/user-attachments/assets/4c6917ac-6f6d-4c60-a595-1ece415b30a2" />
<img width="1670" height="1148" alt="Render 0001" src="https://github.com/user-attachments/assets/1475cdac-8f26-4fba-933d-8a6d2d9820ab" />
<img width="1670" height="1148" alt="Render 0000" src="https://github.com/user-attachments/assets/7e80ddf5-f6a8-41c0-81a8-c1b5fa7760b9" />

***
Hardware Bottleneck Note: Computing 20 eigenstates at 256^3 resolution requires extended Krylov subspace allocations (~22 GB combined VRAM/RAM footprint). Because the memory allocation exceeds the 12 GB VRAM limit of my RTX 3060, data is spilled over the PCIe bus to system RAM, making execution memory-bandwidth bound (~32–40 GB/s PCIe vs. 360 GB/s VRAM).For faster development iterations on 12 GB GPUs, using smaller grids or calculating fewer eigenvalues (num_eigenvalues) with float32 precision allows the entire problem to fit natively within VRAM, dramatically reducing compute times.
