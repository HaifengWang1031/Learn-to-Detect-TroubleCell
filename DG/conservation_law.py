import numpy as np
from numpy.core.fromnumeric import repeat
import scipy.integrate as integrate
from utils import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy
from tqdm import trange


class CL_Solver:
    """
    u_t + f(u)_x = 0, x \in space_interval, t \in [0,final_T]
    u(x,0) = u_0,
    with periodic boundary condition,

    The flux term can choose from
    1) scale linear advcection f(u) = cu
    2) Burgers f(u) = u^2/2
    3) Buckley-Leverett f(u) = u^2/(u^2+0.5*(1-u)^2)
    """
    def __init__(self,basis,basis_order=2):

        self.N = basis_order
        self.basis,self.Dbasis = basis(self.N)

        self.Mass_Matrix()
        self.operator = self.RHS_operator


    def Mass_Matrix(self):
        """
        mass_matrix = delta_x/2 * int_{-1,1} phi_i*phi_j dx
        """
        self.mass_matrix = np.empty((self.N,self.N))
        for n1 in range(self.N):
            for n2 in range(self.N):
                self.mass_matrix[n1,n2] = integrate.quad(lambda x:self.basis[n1](x)*self.basis[n2](x),-1,1)[0]

    def RHS_operator(self,weights):
        # Here, we use Lax-Friedrichs flux:f(u-,u+) = 1/2(f(u-)-f(u+)-\alpha(u+ - u-))

        # transfer the weight to local func
        weights = np.reshape(weights,(self.K,self.N))
        local_func = lambda x,i:sum([weights[i][j]*self.basis[j](x) for j in range(self.N)])

        # local flux 
        f_u = lambda x,i:self.flux(local_func(x,i))

        result = np.zeros((self.N,self.K))
        for e in range(self.K):
            for i in range(self.N):
                #1.numerical flux
                num_flux_r = 1/2*(f_u(1,e) + f_u(-1,e+1 if e<self.K-1 else 0) \
                            - self.max_dflux*(local_func(-1,e+1 if e<self.K-1 else 0) - local_func(1,e)))*self.basis[i](1)
                num_flux_l = 1/2*(f_u(1,e-1) + f_u(-1,e) \
                            - self.max_dflux*(local_func(-1,e) - local_func(1,e-1)))*self.basis[i](-1)
                #2.RHS_interger
                rhs_interger = integrate.quad(lambda x:f_u(x,e)*self.Dbasis[i](x),-1,1)[0]
                result[i][e] = rhs_interger + num_flux_l - num_flux_r
            result[:,e] = np.linalg.solve(self.mass_matrix,result[:,e])/(self.delta_x[e]/2)
        return np.reshape(result,(-1,1),"F")

    def Limiter(self,BasisWeights,slope_limiter):
        cell_quantites = self.limiter_quantites(BasisWeights)
        cell_indicator = self.trouble_cell_indicator(cell_quantites,slope_limiter)
        weights = self.poly_reconstruction(BasisWeights,cell_indicator,cell_quantites,slope_limiter)

        ReconstructedWeight = weights.reshape(-1,1)
        return ReconstructedWeight

    def limiter_quantites(self,BasisWeights):
        weights = np.reshape(BasisWeights,(self.K,self.N))
        local_func = lambda x,i:sum([weights[i][j]*self.basis[j](x) for j in range(self.N)])
        interface_value = np.empty((self.K,2))
        for e in range(self.K):
            interface_value[e,0] = local_func(-1,e)
            interface_value[e,1] = local_func(1,e)

        # 1. Calculate five quantities
        cell_quantites = np.zeros((self.K,5))
        for i in range(self.K):
            # cell average values
            cell_quantites[i,0] = integrate.quad(local_func,-1,1,args = (i,))[0]/2
        for e in range(self.K):
            # forward difference
            cell_quantites[e,1] = cell_quantites[e,0] - cell_quantites[e-1,0]
            cell_quantites[e,3] = cell_quantites[e,0] - interface_value[e,0]
            # backward difference
            cell_quantites[e,2] = cell_quantites[e+1 if e<self.K-1 else 0,0] - cell_quantites[e,0]
            cell_quantites[e,4] = interface_value[e,1] - cell_quantites[e,0]
        cell_quantites = np.concatenate((cell_quantites,interface_value),1)
        return cell_quantites

    def trouble_cell_indicator(self,cell_quantites,slope_limiter):
        # calculate modified cell-interface value
        cell_tilde = np.zeros((self.K,2))
        cell_indicator = np.zeros((self.K,1),dtype=np.bool8)
        for e in range(self.K):
            cell_tilde[e,0] = cell_quantites[e,0] + slope_limiter(cell_quantites[e,3], cell_quantites[e,1], cell_quantites[e,2],self.delta_x[e])
            cell_tilde[e,1] = cell_quantites[e,0] - slope_limiter(cell_quantites[e,4], cell_quantites[e,1], cell_quantites[e,2],self.delta_x[e])
            if np.abs(cell_tilde[e,0]- cell_quantites[e,5])>1e-9 or np.abs(cell_tilde[e,1] -cell_quantites[e,6])>1e-9:
                cell_indicator[e,0] = True
        self.Trouble_Cell = np.concatenate((self.Trouble_Cell,cell_indicator),axis =1)
        return cell_indicator

    def poly_reconstruction(self,BasisWeights,cell_indicator,cell_quantites,slope_limiter):
        weights = np.reshape(BasisWeights,(self.K,self.N))
        # local_Dfunc = lambda x,i:sum([weights[i][j]*self.Dbasis[j](x) for j in range(self.N)])

        for e in range(self.K):
            if cell_indicator[e,0]:
                #step 2. use a suitable limiter to reconstructing the polynomial solution
                weights[e,:] = 0.
                # Baseline Reconstructer
                left_interface_value = cell_quantites[e,0] + slope_limiter(cell_quantites[e,3], cell_quantites[e,1], cell_quantites[e,2],self.delta_x[e],self.delta_x[e])
                right_interface_value = cell_quantites[e,0] - slope_limiter(cell_quantites[e,4], cell_quantites[e,1], cell_quantites[e,2],self.delta_x[e],self.delta_x[e])
                weights[e,0] = (left_interface_value + right_interface_value)/2
                weights[e,1] = (left_interface_value - right_interface_value)/2
                
                # classic MUSCL reconstruction
                # slope = slope_limiter(local_Dfunc(0,e), 
                #     (cell_quantites[e,0] - cell_quantites[e-1,0])/(self.delta_x[e]/2), 
                #     (cell_quantites[e+1 if e<self.K-1 else 0,0] - cell_quantites[e,0])/(self.delta_x[e]/2))
                # weights[e,0] = cell_quantites[e,0] + slope
                # weights[e,1] = slope
        return weights
        

    def reset(self,init_func,flux,space_interval,ele_num = 32):
        # space partition
        self.space_interval = space_interval
        self.K = ele_num
        self.x_node = np.linspace(*self.space_interval,num=self.K+1)
        self.delta_x = np.diff(self.x_node)
        self.x_h = np.vstack([self.x_node[0:-1],self.x_node[1:]]).T

        # transfer the initial function to basis weights
        ExactRHS = np.zeros((self.N,self.K))
        for n in range(self.N):
            for k in range(self.K):
                ExactRHS[n][k] = integrate.quad(lambda x:init_func(x)*self.basis[n]((2*x - (self.x_h[k][0]+self.x_h[k][1]))/self.delta_x[k]),self.x_h[k][0],self.x_h[k][1])[0]/(self.delta_x[k]/2)

        BasisWeights = np.linalg.solve(self.mass_matrix,ExactRHS)
        BasisWeights = np.reshape(BasisWeights,(self.N*self.K,1),order="F")
        self.BasisWeights = BasisWeights

        self.WeightContainer = self.BasisWeights

        # set the flux term and calculate alpha
        self.flux = flux
        x = sympy.Symbol("x")
        flux_sym = flux(x)
        dflux = sympy.lambdify(x,sympy.diff(flux_sym,x),"numpy")
        self.max_dflux = np.max(dflux(self.x_node))

        self.Trouble_Cell = np.zeros((self.K,1),dtype=np.bool8)

    def step(self,delta_t,evolution_method="Euler"):
        assert evolution_method in ["RK3","RK2","Euler"]
        if evolution_method == "RK3":
            # TDV-RK3 method
            w1 = self.BasisWeights + self.operator(self.BasisWeights)*delta_t
            w2 = 3/4*self.BasisWeights + 1/4*(w1 + self.operator(w1)*delta_t)
            BasisWeights = 1/3*self.BasisWeights + 2/3*(w2 +self.operator(w2)*delta_t)
        elif evolution_method == "RK2":
            # TDV-RK2 method
            w1 = self.BasisWeights + self.operator(self.BasisWeights)*delta_t
            BasisWeights = 1/2*self.BasisWeights + 1/2*(w1 +self.operator(w1)*delta_t)
        elif evolution_method == "Euler":
            # Euler method
            BasisWeights = self.BasisWeights + self.operator(self.BasisWeights)*delta_t
        return BasisWeights

    def solve(self,final_time = 1, cfl=0.1,evolution_method="Euler",use_limiter =False,slope_limiter = minmod_limiter,render=False):
        delta_t = cfl*np.max(self.delta_x)
        self.delta_t = delta_t
        n = np.ceil(final_time/delta_t).astype(int)
        for i in trange(n):
            BasisWeights = self.step(delta_t,evolution_method)
            if use_limiter:
                BasisWeights = self.Limiter(BasisWeights,slope_limiter)
            self.BasisWeights = BasisWeights
            self.WeightContainer =  np.concatenate((self.WeightContainer,self.BasisWeights),axis=1)

        self.get_node_value()
        if render:
            # self.draw_whole()
            self.render()
        
    def get_node_value(self):
        node_per_ele = 10
        for t in range(0,len(self.WeightContainer[0])):
            for e,i in enumerate(range(0,len(self.WeightContainer[:,t]),self.N)):
                x_e = np.linspace(self.x_h[e][0],self.x_h[e][1],
                                  node_per_ele if e<self.K-1 else node_per_ele+1,
                                  endpoint=False if e <self.K-1 else True )
                
                result_local = np.zeros_like(x_e)
                for j in range(self.N):
                    result_local +=self.WeightContainer[:,t][i+j]*self.basis[j]((2*x_e - (self.x_h[e][0]+self.x_h[e][1]))/self.delta_x[e])
                
                if e == 0:
                    current_value = result_local
                else:
                    current_value = np.append(current_value,result_local)
                
                if t == 0:
                    if e==0:
                        node_point = x_e
                    else:
                        node_point = np.append(node_point,x_e)
            if t == 0:
                solu_value = current_value
            else:
                solu_value = np.vstack((solu_value,current_value))
        return node_point,solu_value

    def render(self):
        node_point,solu_value = self.get_node_value()

        fig,ax = plt.subplots() 
        line, = ax.plot(node_point,solu_value[0])

        def init():
            ax.set_xlim(self.space_interval[0],self.space_interval[1])
            ax.set_ylim(np.min(solu_value)-0.1,np.max(solu_value)+0.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            return line,

        def update(t):
            line.set_ydata(solu_value[t,:])
            ax.set_title(f"t = {self.delta_t*t:.4f}s")
            return line,
        
        ani = animation.FuncAnimation(fig, update,range(1,len(solu_value)), interval=100, init_func = init,repeat = False)
        plt.show()

    def draw_whole(self):
        n = len(self.WeightContainer[0,:])
        plt.ion()
        for i in range(n):
            self.draw_step(self.WeightContainer[:,i])
            plt.pause(self.delta_t)
            plt.clf()
        plt.ioff()

    def draw_step(self,weights):
        for e,i in enumerate(range(0,len(weights),self.N)):
            x_e = np.linspace(self.x_h[e][0],self.x_h[e][1])
            result_local = np.zeros_like(x_e)
            for j in range(self.N):
                result_local += weights[i+j]*self.basis[j]((2*x_e - (self.x_h[e][0]+self.x_h[e][1]))/self.delta_x[e])
            plt.plot(x_e,result_local)
            plt.draw()
        
        
if __name__ == "__main__":
    #config
    basis = legendre_basis
    basis_order = 4
    init_func = sine_wave
    space_interval = [0,1]
    flux = lambda x:x**2/2
    ele_num = 100
    final_time = 1
    cfl = 0.1
    evolution_method = "RK3"
    use_limiter = True
    slope_limiter = minmod_limiter
    render = True

    solver = CL_Solver(basis,basis_order)
    solver.reset(init_func, flux ,space_interval,ele_num)
    solver.solve(final_time,cfl,evolution_method,use_limiter,slope_limiter,render)