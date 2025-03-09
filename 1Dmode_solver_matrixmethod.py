# here I solve the 1d waveguide using matrixes:)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import sparse
matplotlib.use('TkAgg')

lmbd=1.55 #um 
n_sub=1.0
n_guide=3.45
n_clad=1.5
t_core=0.22 #um
t_wg=0.22
sim_space= 2 #um => total dimension of the simulation 
step=0.001 #um 
k=2*np.pi/lmbd


def ref_index(x:float)->float:
    index_value=None
    if -t_core/2<=x<=t_core/2:
        index_value=n_guide
    elif x<-t_core/2:
        index_value=n_sub
    elif x>t_core/2:
        index_value=n_clad
    return index_value

# def ref_index(x:float)->float:
#     index_value=n_sub
#     #print(x)
#     if -t_core/2-t_wg<x<=-t_core/2:
#         index_value=n_guide
#     elif -t_core/2<x<=t_core/2:
#         index_value=n_clad
#     elif t_core/2<x<=t_core/2+t_wg:
#         index_value=n_guide
#     return index_value


# def F(x:float,n_test:float)-> float:
#     k=2*np.pi/lmbd
#     Beta=n_test*k
#     return Beta**2-k**2*ref_index(x)**2

def build_diff_matrix(x):
    N_components=len(x)
    A=np.zeros(shape=(N_components,N_components))
    for i in range(0,N_components-1):
        A[i,i]=-2+ref_index(x[i])**2*k**2*step**2
        A[i,i+1]=1
        A[i+1,i]=1
    A[N_components-1,N_components-1]=-2+ref_index(x[N_components-1])**2*step**2*k**2
    #A=sparse.csr_matrix(A)
    return A

def solve_eigen(n_start,x):
    A=build_diff_matrix(x)
    #phi_vector=np.random.random(len(x))
    phi_vector=np.ones(len(x))
    #print(A)
    print(k)
    diff_min=0.0001
    diff=2
    p=step**2*n_start**2*k**2
    I=np.identity(len(x))
    #while diff<diff_min:
    print("start iterations:",flush=True)
    for i in range(20):
        new_matrix=A-p*I
        #print(new_matrix[0,0])
        phi_vector=np.linalg.solve(new_matrix,phi_vector)
        #print(A-p*I)
        #print(phi_vector)
        c=np.argmax(abs(phi_vector))
        diff=1/phi_vector[c]
        phi_vector=phi_vector*diff
        eigen=p+diff
    plt.plot(phi_vector)
        #print(eigen,flush=True)
    neff=np.sqrt(eigen/k**2/step**2)
    print(neff)

def main():
    x=np.arange(-sim_space/2,sim_space/2,step)
    N_step=len(x)
    solve_eigen(3.1,x)
    plt.show()

if __name__=="__main__":
    main()