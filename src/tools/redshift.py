import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import math as m

arcmin = 10.0
deg = arcmin/60/360
rad = 2*m.pi*deg
c = 299792458.0

H0 = 70*1000.0
M = 0.25
L = 0.75

print(f'{arcmin} arcmin;')
print(f'{M} OMEGAm; {L} OMEGAl; {H0/1000} Ho km/Mpc/s')
print('\n')

def co_moving(z, M, L, c, H0):
    func = lambda z: (M*(1+z)**3 + L)**(-1/2)
    return (c/H0)*integrate.quad(func, 0, z)[0] 

def ang_dia_dis(Dm, z):
    return (Dm/(1 + z))

def transverse_x(Da, theta):
    return theta*Da


for z in [0.04, 0.5, 1.0, 2.0]:
    Dc = co_moving(z, M, L, c, H0)
    Dc_str = 'Dc: {} Mpc/h'.format(np.round(Dc, 1))

    Dm = Dc

    Da = ang_dia_dis(Dm, z)
    Da_str = 'Da: {} Mpc/h'.format(np.round(Da, 1))

    x_one = transverse_x(Da, 2*m.pi/360)
    x_one_str = '1 degree: {} Mpc/h proper\n'.format(np.round(x_one, 1))

    px_one_co = Dm*rad
    px_one_co_str = '1px: {} Mpc/h co-moving'.format(np.round(px_one_co, 1))
    px_256_co_str = '256 px: {} Mpc/h co-moving\n'.format(np.round(256*px_one_co, 1))

    px_one = transverse_x(Da, rad)
    px_one_str = '1px: {} Mpc/h proper'.format(np.round(px_one, 1))
    px_256_str = '256px: {} Mpc/h proper'.format(np.round(256*px_one, 1))


    info = [Dc_str, Da_str, x_one_str, 
            px_one_co_str,
            px_256_co_str,
            px_one_str,
            px_256_str,
            ]
    print('redshift {}'.format(z))
    for string in info:
        print(2*' ' + string)
    print('\n')
