# here I solve the 1d waveguide using matrixes:)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import sparse
import taichi as ti
ti.init(arch=ti.cpu)
matplotlib.use('TkAgg')

lmbd=1.55 #um 
n_sub=1.0
n_guide=3.45
n_clad=1.5
t_core=0.22 #um
t_wg=0.22
sim_space= 2#um => total dimension of the simulation 
step=0.001 #um 
k=2*np.pi/lmbd


@ti.func
def ref_index(x:ti.types.f64)->ti.types.f64:
    index_value=0.000
    #print(x)
    if -t_core/2.0<=x<=t_core/2.0:
        index_value=n_guide
    elif x<-t_core/2:
        index_value=n_sub
    elif x>t_core/2:
        index_value=n_clad
    return index_value

# @ti.func
# def ref_index(x:ti.types.f64)->ti.types.f64:
#     index_value=n_sub
#     #print(x)
#     if -t_core/2-t_wg<x<=-t_core/2:
#         index_value=n_guide
#     elif -t_core/2<x<=t_core/2:
#         index_value=n_clad
#     elif t_core/2<x<=t_core/2+t_wg:
#         index_value=n_guide
#     return index_value



res=5
@ti.kernel
def build_diff_matrix(A: ti.types.sparse_matrix_builder(),x:ti.types.ndarray(),N_components:ti.types.int32,k:ti.types.f64):
    print(k)
    for i in ti.ndrange(N_components-1):
        A[i,i]+=-2+ref_index(x[i])**2*k**2*step**2
        A[i,i+1]+=1.0
        A[i+1,i]+=1.0
        #print(i)
    #A[N_components,N_components]+=-2+ref_index(x[N_components-1])**2*step**2*k**2
    #return A

@ti.kernel
def build_I(A:ti.types.sparse_matrix_builder(),b:ti.types.ndarray(),N:ti.types.int32):
    for i in ti.ndrange(N):
        A[i,i]+=1.0
        b[i]+=0.5

def solve_eigen(n_start,x):
    
    #build test matrix
    K2=ti.linalg.SparseMatrixBuilder(len(x),len(x),max_num_triplets=len(x)*52,dtype=ti.f64)
    build_diff_matrix(K2,x,len(x),k)
    A=K2.build()
    #build identity
    I_b=ti.linalg.SparseMatrixBuilder(len(x),len(x),max_num_triplets=len(x)*5,dtype=ti.f64)
    phi_vector=ti.ndarray(ti.types.f64,shape=len(x))
    build_I(I_b,phi_vector,len(x))
    I=I_b.build()
    #print(I)
    #print(phi_vector.to_numpy())
    #print(k)
    #phi_vector=np.random.random(len(x))
    #print(A)
    diff_min=step # placed equal to the step size 
    diff=2
    p=step**2*n_start**2*k**2
    neff_old=n_start

    print("start iterations:",flush=True)
    for i in range(1000):
        solver=ti.linalg.SparseSolver(solver_type="LDLT",dtype=ti.f64,ordering="COLAMD")
        new_matrix=A-p*I
        solver.compute(new_matrix)
        phi_vector=solver.solve(phi_vector)
        calculation_vector=phi_vector.to_numpy()
        
        index_max=np.argmax(abs(calculation_vector))
        diff=1/calculation_vector[index_max]

        phi_vector.from_numpy(calculation_vector*diff)
        # processing for calculating neff
        eigen=p+diff
        neff_factor=k**2*step**2
        neff_new=np.sqrt(eigen/neff_factor)
        neff_diff=neff_new-neff_old
        neff_old=neff_new
        print(neff_diff,flush=True)
        if abs(neff_diff)<diff_min:
            print(i,flush=True)
            break
    #print(diff)
    plt.plot(phi_vector)
        
    neff=np.sqrt(eigen/k**2/step**2)
    print(neff)

def main():
    x=np.arange(-sim_space/2,sim_space/2,step)
    #print(x)
    N_step=len(x)
    solve_eigen(2.5,x)
    plt.show()

if __name__=="__main__":
    main()