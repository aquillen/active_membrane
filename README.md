# active_membrane
We use a finite element method to simulate an active membrane. 
We integrate the system of partial differential equation 

$$ \partial_t {\bf u} = -\pi_k {\bf u} + \nabla \sigma + f(p) {\bf p} $$
$$ \partial_t {\bf p} = - {\bf p} + \partial_t {\bf u} $$
where $\sigma$ is the stress tensor and 
$$ f(p) = \frac{A}{1 + p/p_*} $$

Here ${\bf u}, {\bf p} \in {\mathbb R}^2$ and ${\bf u}({\bf x}, t)$, ${\bf p}({\bf x},t)$ with ${\bf x} \in \Omega$ and $\Omega$ is a two-dimensional domain.   The real parameters $A, \pi_k, p_* >0$. 

The system represents an active membrane.  The system exhibits two limit cycles.  One rotates in the opposite direction of the other. 

The image below shows an integration on a circular domain with $A=3$, $\pi_k = 0.1$, $p_*=0.5$ on a circular domain with zero Dirchlet boundary condition. 

<img src="https://github.com/aquillen/active_membrane/blob/main/Circ_v003_tryvel_625c.png" alt="">

![For a video](https://raw.githubusercontent.com/aquillen/active_membrane/main/video5_short.mp4)

The notebooks use finite element package NGsolve and subroutines in the file bio_mem_helpers.py 

Notebooks:

+ Circ_v002.ipynb Circular domain Neumann Boundary (denoted Circ-N in manuscript)
+ Circ_v003.ipynb Circular domain Dirichlet Boundary  (denoted Circ-D in manuscript)
+ Rec_v002.ipynb Rectangular domain, mixed boundary condition (denoted Rec-Small in manuscript)
+ Rec_v003.ipynb Rectangular domain, mixed boundary condition (denoted Rec-Mid in manuscript)
+ Ann_v003.ipynb Annular domain, mixed boundary condition (denoted Ann in manuscript)

+ Dispersions.ipynb Plots roots of dispersion relations and bifurcation plot

Subroutines:
+ bio_mem_helpers.py

The integrator is Crank-Nicolson for the elastic term and a first order correction afterwards to take into account the non-linear terms in the PDE system. 
