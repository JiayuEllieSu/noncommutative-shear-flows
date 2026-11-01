# Noncommutative Shear Flows and Chaotic Advection: An Area-Preserving Map Model
A computational theory project for PHY630 (Fluid Mechanics).

## Overview

This project explores how simple horizontal and vertical shear flows can combine to produce complicated tracer motion. The focus is on the relationship between incompressibility, area-preserving transformations, noncommutative matrix multiplication, and material stretching.

The computational model uses idealized two-dimensional shear maps and periodic boundaries. It is a model of passive-particle advection, not a full Navier–Stokes simulation. Stretching and rearrangement of tracers will be distinguished from molecular mixing, which also involves diffusion.

## Planned work

1. Derive shear transformations from incompressible velocity fields and review their group properties.
2. Implement horizontal, vertical, and alternating shears and visualize tracer-particle motion.
3. Compare the order of transformations, estimate material stretching, and relate the results to the eigenvalues of the combined map.
4. Discuss the physical interpretation and limitations of the model.

## References

- H. Aref, “Stirring by Chaotic Advection,” *Journal of Fluid Mechanics* (1984). https://doi.org/10.1017/S0022112084001233
- J.-L. Thiffeault and S. Childress, “Chaotic Mixing in a Torus Map,” *Chaos* **13** (2003), 502–507. https://doi.org/10.1063/1.1568833
- J. M. Ottino, “Mixing, Chaotic Advection, and Turbulence,” *Annual Review of Fluid Mechanics* **22** (1990), 207–253. https://doi.org/10.1146/annurev.fl.22.010190.001231
