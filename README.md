# active_membrane
We use a finite element method to simulate an active membrane. 
We integrate the system of partial differential equation 

$$ \partial_t {\bf u} = -\pi_k {\bf u} + \nabla \sigma + f(p) {\bf p} $$
$$ \partial_t {\bf p} = - {\bf p} + \partial_t {\bf u} $$
where $\sigma$ is the stress tensor and 
$$ f(p) = \frac{A}{1 + p/p_*} $$

Here ${\bf u}, {\bf p} \in {\mathbb R}^2$ and ${\bf u}({\bf x}, t)$, ${\bf p}({\bf x},t)$ with ${\bf x} \in \Omega$ and $\Omega$ is a two-dimensional domain.   The real parameters $A, \pi_k, p_* >0$. 

The system represents an active membrane.  The system exhibits two counterrotating limit cycles. 

![](https://github.com/aquillen/active_membrane/blob/main/bio_v006_vel.gif)

The notebooks use NGsolve and subroutines in bio_mem_helpers.py. 

The notebooks if they start with Circ are on circular domains. 
