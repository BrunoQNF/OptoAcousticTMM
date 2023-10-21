# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from materials import GaAs
import numpy as np


def raman_analytic(z, a, b, c, d, k,k2):
    i1 = (np.abs(a)**2 + np.abs(b)**2)*(c*np.exp(1j*k2*z) + d*np.exp(-1j*k2*z) - c - d)
    i2 = 2*(np.real(a)*np.real(b)+np.imag(a)*np.imag(b))*(np.exp(-1j*k2*z)*k2*((d+c*np.exp(2j*k2*z))*k2*np.cos(2*k*z)+ 2j*(d-c*np.exp(2j*k2*z))*k*np.sin(2*k*z))-(c+d)*k2**2)/(-4*k**2+k2**2)
    i3 = 2*(np.real(a)*np.imag(b)- np.imag(a)*np.real(b))*(np.exp(-1j*k2*z)*k2*(-2j*(d-c*np.exp(2j*k2*z))*k*np.cos(2*k*z)+ (d+c*np.exp(2j*k2*z))*k2*np.sin(2*k*z))-2j*(c-d)*k2*k)/(-4*k**2+k2**2)
    return i1 + i2 + i3

def raman_cross_section(opt,reso):
    inds = opt.photoelastic_index()
    opt.calc_complex_amplitudes()
    reso.calc_complex_amplitudes()
    raman_total = 0 + 0*1j
    dist_integral = np.array(opt.ds*opt.p[0] + [opt.spacer[1]] + opt.ds[::-1]*opt.p[1])[inds-1]
    k_optic , k_acoustic = opt.ks[inds[0]-1],reso.ks[inds[0]-1]
    for i,val in enumerate(inds):
        raman_total += raman_analytic(dist_integral[i],opt.Es[val][0],opt.Es[val][1],reso.Es[val][0],reso.Es[val][1],k_optic,k_acoustic)
    return np.abs(raman_total)**2
    
    
    
    