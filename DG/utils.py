import numpy as np
import sympy

def legendre_basis(deg=2):
    basis = [
        lambda x:np.ones_like(x),
        lambda x:x,
        lambda x:(3*x**2-1)/2,
        lambda x:(5*x**3-3*x)/2,
        lambda x:(35*x**4-30*x**2+3)/8,
        lambda x:(63*x**5-70*x**3+15*x)/8
    ]
    Dbasis = [
        lambda x:np.zeros_like(x),
        lambda x:np.ones_like(x),
        lambda x:3*x,
        lambda x:(5*3*x**2-3)/2,
        lambda x:(35*4*x**3-30*2*x)/8,
        lambda x:(63*5*x**4-70*2*x**2+15)/8
    ]
    return basis[:deg],Dbasis[:deg]

def baseline_basis(deg = 2):
    nodes = np.linspace(-1,1,deg)
    x = sympy.Symbol("x")
    def N_basis(x,x_j):
        result = 1
        for node in nodes:
            if node != x_j:
                result *= (x-node)/(x_j-node)
        return result
    basis = []
    for j in nodes:
        basis.append(N_basis(x,j))
    Dbasis = [sympy.diff(fun,x) for fun in basis]
    
    return list(map(lambda func:sympy.lambdify(x,func,"numpy"),basis)),list(map(lambda func:sympy.lambdify(x,func,"numpy"),Dbasis))

# Limiter
def minmod_limiter(a,b,c,*args):
    if np.sign(a) == np.sign(b) == np.sign(c):
        return np.sign(a)*np.min([np.abs(a),np.abs(b),np.abs(c)])
    else:
        return 0

def TVB_limiter(a,b,c,h,M):
    if np.abs(a) <= M*h**2:
        return a
    else:
        return minmod_limiter(a, b, c)

# initial wave for Linear advection equation
def sine_wave(x):
    # x in [0,1]
    return np.sin(10*np.pi*x)

def multi_wave(x):
    # x in [0,1.4]
    if 0.2 < x <=0.3:
        return 10*(x-0.2)
    elif 0.3 < x <= 0.4:
        return 10*(0.4-x)
    elif 0.6< x <=0.8:
        return np.ones_like(x)
    elif 1< x <=1.2:
        return 100*(x-1)*(1.2-x)
    else:
        return np.zeros_like(x)

# initial wave for Burgers equation
def shock_collision(x):
    # x in [0,1]
    if x<=0.2:
        return 5*np.ones_like(x)
    elif 0.2<x<=0.4:
        return 2*np.ones_like(x)
    elif 0.4<x<=0.6:
        return 0*np.ones_like(x)
    elif x>0.6:
        return -2*np.ones_like(x)


def compound_wave(x):
    # x in [-4,4]
    if x >= 1 or x <= -1:
        return np.sin(np.pi*x)
    elif -1<x<= -0.5:
        return 3*np.ones_like(x)
    elif -0.5<x<=0:
        return 1*np.ones_like(x)
    elif 0<x<= 0.5:
        return 3*np.ones_like(x)
    elif 0.5<x<=1:
        return 2*np.ones_like(x)

# initial wave of Buckley Levertt equation
def b_l_initial(x):
    if x>= 0.5:
        return 0.95*np.ones_like(x)
    elif x< 0.5:
        return 0.1*np.ones_like(x)


def transfer_wave(init_func,interval,velocity):
    assert interval[1] > interval[0]
    len_interval = interval[1] - interval[0]

    # periodic expension
    @eleMapping
    def Periodic_func(x):
        if interval[0]<x<=interval[1]:
            return init_func(x)
        elif x > interval[1]:
            return Periodic_func(x-len_interval)
        elif x <= interval[0]:
            return Periodic_func(x+len_interval)
    return lambda x,t:Periodic_func(x-velocity*t)

def eleMapping(n_func):
    def wrapTheFunction(x):
        if x.ndim == 0:
            return n_func(x)
        elif x.ndim == 1:
            result = np.empty_like(x)
            for i,n in enumerate(x):
                result[i] = n_func(n)
            return result
    return wrapTheFunction

if __name__ == "__main__":
    pass