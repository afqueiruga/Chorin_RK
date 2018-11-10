---
layout: post
title: "Generalizing Chorin's Method for Higher-Order Runge Kuttas"
date: 2018-3-1
categories: simulation
---

## NOTE: This code hasn't passed benchmarking yet. Don't trust it!

## Overview

Chorin's method is one of the standard ways of solving incompressible Navier Stokes.
The basic idea is to explicitly compute a trial step for the velocity, then compute the pressure that corrects the divergence of the velocity, and finally apply that pressure gradient to copmute the final velocity.
[The FEniCS tutorial explains it well with an FEM implementation.](https://fenicsproject.org/olddocs/dolfin/1.6.0/python/demo/documented/navier-stokes/python/documentation.html)
However, the standard phrasing of the method assumes using a forward Euler time discretization, $\partial_t u \approx (u-u _0)/\Delta t$.
It's good enough for solving Navier Stokes efficiently, but what if we are trying to couple the fluid with an ODE that has different stability requirements?

How do we rephrase the method as a general Runge Kutta?

## Formulation

The partial differential algebraic equation for Navier Stokes is
\begin{equation}
\partial_t u + \nabla u \cdot u - \mathrm{Re}^{-1} \nabla^2 u + \nabla p = f
\end{equation}
subject to the constraint
\begin{equation}
\nabla\cdot u = 0.
\end{equation}
The standard derivation of Chorin's method shoves in the forward Euler approximation for the time rate immediately.
I like to do that as the final step to make it easier to cycle through time steppers.

Let us define a forcing term $r$:
\begin{equation}
r = f - \nabla u\cdot u + \mathrm{Re}^{-1} \nabla \cdot \nabla u.
\end{equation}
This lets us rewrite the ODE as
\begin{equation}
\partial _t u = r - \nabla p
\end{equation}
which is still subject to the same constraint as before.
We're allowed to take the time derivative of the constraint, and exchanging with the spatial divergence gives us a new equation,
\begin{equation}
\partial _t \nabla \cdot u = 0 \quad \rightarrow \quad
\nabla \cdot \partial _t u = 0 \quad \rightarrow \quad
\nabla \cdot (r-\nabla p) = 0
\end{equation}
We can use this equation to solve for $p$ initially for $r$, and then apply our time integrator to $\partial _t u$.

Now we want to phrase this in a variational formulation to weaken the derivations to apply the finite element method.
The key to this reconsideration is that we recast the trial velocity step as a projection of the equation for $r$ onto the velocity basis functions.
Let $v,\delta v \in \mathcal{V}$ be functions and test functions in a suitable space for velocities, and $p,\delta p \in \mathcal{P}$ be the same for the pressures.

How do we phrase this as an explicit integrator?

1. Project $r(t)$ onto $\mathcal{V}$ by solving
\begin{equation}
(\delta u, r) = (\delta u, f) - (\delta u, \nabla u \cdot u) - \mathrm{Re}^{-1} (\nabla \delta u, \nabla u)
\end{equation}
2. Solve for $p(t+h)$ using $r$:
\begin{equation}
(\nabla \delta p, \nabla p) = (\delta p, -\nabla\cdot r)
\end{equation}
3. Apply the time stepper to the ODE:
\begin{equation}
(\delta u, \partial_t u) = (\delta u, r) - (\delta u, \nabla p)
\end{equation}
which comes out to the discrete system that can integrated normally:
\begin{equation}
\mathbf{M} \dot{\mathbf{u}} = \mathbf{R}(t)
\end{equation}

The process is phrased as a projection step as an optimization; it would be possible to skip 1 and have steps 2 and 3 reperform the calculations inside of two different element assembly routines.
However, the force coupling term $(\delta u,f)$ may be a very complicated interaction between two different numerical methods that we do not want to repeat.
For example, as I'm writing this I'm implementing a particle-fluid code where that term is a discrete summation of point-integrals using my [FEniCS_Particles](https://github.com/afqueiruga/FEniCS_Particles) library.

## Implementation in FEniCS

We use the standard Taylor-Hood elements (quadratic velocities and linear pressures).
The forms needed to perform the above steps look as follows in the FEniCS UFL:
```Python
f_v_M = inner(tu,Du)*dx
f_r_proj = - inner(tu,dot(grad(u),u))*dx \
       - mu*inner(grad(tu),grad(u))*dx
f_p_M = inner(tp,Dp)*dx
f_p_r = inner(tp,-div(r))*dx
f_v_dot = inner(tu, r - grad(p))*dx
```
This projection step requires us to implement a new class. We merge
the step into the implicit pressure 
```Python
class RK_field_chorin_pressure(pyrk.RKbase.RK_field_dolfin):
def __init__(self, r, p, f_v_M, f_r_proj, f_p_K, f_p_R, bcs=None, **kwargs):
    self.r, self.p = r, p
    self.f_r_proj = f_r_proj
    self.f_p_R = f_p_R
    self.bcs = bcs
    self.K_p = assemble(f_p_K)
    self.M_v = assemble(f_v_M)
    pyrk.RKbase.RK_field_dolfin.__init__(self, 0, [p.vector()], None, **kwargs)
def sys(self,time,tang=False):
    solve(self.M_v, self.r.vector(), assemble(self.f_r_proj))
    R = -assemble(self.f_p_R) + self.K_p*self.p.vector()
    return [R, self.K_p] if tang else R
```
The big addition to this class from the standard `RK_field_fenics` class is that it performs the $r$ projection step at the first line of `sys()` before return $\mathbf{R}$ and $\mathbf{K}$.

We can then initialize this special `RK_field` class and the standard `RK_field_fenics` class to have two modules representing the pressure prediction and the velocity ODE, and insert them into an explicit Runge Kutta time stepper object with any tableau we desire:
```Python
rkf_p = RK_field_chorin_pressure(r,p,f_v_M, f_r_proj, f_p_K,f_p_r, bcs_p)
rkf_v = RK_field_fenics(1, [ u ], f_v_M, f_v_dot, [], bcs_v )
Tfinal = 0.01
DeltaT = Tfinal/100.0
step = pyrk.exRK.exRK(DeltaT, pyrk.exRK.exRK_table['RK4'], [rkf_p, rkf_v] )
```

## Order of accuracy
