from fenics import *
import afqsrungekutta as pyrk
from afqsrungekutta.rkfenics import RK_field_fenics
from afqsfenicsutil import write_vtk_f

krylov_method = "gmres"

L = 1.0
h = 1.0
Dp = 1.0
mesh = RectangleMesh(Point(-L/2.0, -h/2.0),Point(L/2.0,h/2.0) ,40,40, "right/left")
boundfunc = MeshFunction("size_t",mesh, mesh.topology().dim()-1)
boundfunc.set_all(0)
top = CompiledSubDomain(" x[1]>h/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
top.mark(boundfunc,1)
bot = CompiledSubDomain("-x[1]>h/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
bot.mark(boundfunc,2)
left = CompiledSubDomain("-x[0]>L/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
left.mark(boundfunc,3)
right = CompiledSubDomain("x[0]>L/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
right.mark(boundfunc,4)


# Select dimension-dependent aspects
cell = mesh.ufl_cell()
gdim = mesh.geometry().dim()
if gdim == 2:
    zeroV = Constant((0.,0.))
else:
    zeroV = Constant((0.,0.,0.))

V2 = VectorFunctionSpace(mesh, "CG",2)
S1 = FunctionSpace(mesh, 'CG',1)
u = Function(V2)
r = Function(V2)
tu = TestFunction(V2)
Du = TrialFunction(V2)
p = Function(S1)
tp = TestFunction(S1)
Dp = TrialFunction(S1)

# Set up the boundary conditions
noslip = Constant((0.0, 0.0))
drive = Constant((1.0, 0.0))
bcs_v = [
    DirichletBC(V2,noslip,boundfunc,1),
    DirichletBC(V2,noslip,boundfunc,2),
    DirichletBC(V2,noslip,boundfunc,3),
    DirichletBC(V2,noslip,boundfunc,4)
]
bcs_p = [
    DirichletBC(S1,0,boundfunc,1),
]
# properties
mu = 1/100.0

f_v_M = inner(tu,Du)*dx
f_r_proj = - inner(tu,dot(grad(u),u))*dx \
           - mu*inner(grad(tu),grad(u))*dx
f_p_K = inner(grad(tp),grad(Dp))*dx
f_p_r = inner(tp,-div(r))*dx
f_v_dot = inner(tu, r - grad(p))*dx


class RK_field_chorin_pressure(pyrk.RKbase.RK_field_dolfin):
    """
    Just give this class FEniCS forms!
    """
    def __init__(self, r, p, f_v_M, f_r_proj, f_p_K, f_p_R, bcs=None, **kwargs):
        self.r = r
        self.p = p
        self.f_r_proj = f_r_proj
        self.f_p_R = f_p_R
        self.bcs = bcs
        self.K_p = assemble(f_p_K)
        self.M_v = assemble(f_v_M)
        pyrk.RKbase.RK_field_dolfin.__init__(self, 0, [p.vector()], None, **kwargs)
        
    def sys(self,time,tang=False):
        solve(self.M_v, self.r.vector(), assemble(self.f_r_proj))
        R = assemble(self.f_p_R) + self.K_p*self.p.vector()
        return [R, self.K_p] if tang else R
    
    def bcapp(self,K,R,t,hold=False):
        if self.bcs is not None:
            for bc in self.bcs:
                if K is not None:
                    bc.apply(K)
                if R is not None:
                    bc.apply(R)

rkf_p = RK_field_chorin_pressure(r,p,f_v_M, f_r_proj, f_p_K,f_p_r, bcs_p)
rkf_v = RK_field_fenics(1, [ u ], f_v_M, f_v_dot, [], bcs_v )

onum = 0
def output():
    global onum
    write_vtk_f("outs/viz_{0}.vtk".format(onum),
                mesh,{"u":u, "p":p, "r":r},None)
    onum+=1
    
Tnow = 0.0
Tfinal = 0.01
DeltaT = Tfinal/100.0
delta_outp = 1*DeltaT
step = pyrk.exRK.exRK(DeltaT, pyrk.exRK.exRK_table['RK4'], [rkf_p, rkf_v] )

# Initial condition is v is (1,0) at the top
u.interpolate(Expression(("(x[1]-0.5*h)/h","0.0"),h=h, degree=2))
output()
next_outp = delta_outp-1.0e-1
while Tnow < Tfinal:
    step.march()
    if Tnow >= next_outp:
        output()
        next_outp += delta_outp
        print "Wrote at ", Tnow
    Tnow += DeltaT
