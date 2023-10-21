# -*- coding: utf-8 -*-
"""
SuperLattice v 1.01
Author: Bruno Tenorio
v.1.01 permite celdas unidad de mas de 2 layers
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy.linalg as la

class Superlattice:
    
    def __init__(self,maters, ds, wl, entry, exit, p=1, mode='acoustic'):
        self.maters = maters
        self.ds = ds
        self.p = p
        self.wl = wl
        self.entry = entry
        self.exit = exit
        self.mode = mode
        self.Es = [[1+0*1j,0+0*1j]]
        
    
    def matrix_interface(self, ni, nj):
        return np.array([[1 + ni/nj, 1 - ni/nj], [1 - ni/nj, 1 + ni/nj]]) * 0.5
    
    def matrix_prop(self, k_med, d):
        phase = 1j*k_med*d
        return np.array([[np.exp(phase),0],[0,np.exp(-phase)]])
    
    def build_transfer_matrix(self):
        if self.mode == 'acoustic':
            ks = [2*np.pi/(self.wl * mat.c / self.entry.c) for mat in self.maters]
            meds = [mat.Z for mat in self.maters]                   
            med_entry, med_exit = self.entry.Z, self.exit.Z
        elif self.mode == 'optical' :
            ks = [2*np.pi*mat.n/self.wl for mat in self.maters]
            meds = [mat.n for mat in self.maters]
            med_entry, med_exit = self.entry.n, self.exit.n
        matrix_cell = np.identity(2 , dtype='complex')
        for i in range(len(self.maters)-1,0,-1):
            matrix_cell =  self.matrix_interface(meds[i], meds[i-1]) @ self.matrix_prop(ks[i], -self.ds[i]) @ matrix_cell
        
        matrix_cell = self.matrix_prop(ks[0], -self.ds[0]) @ matrix_cell
        if self.p == 1:
            return self.matrix_interface(meds[0], med_entry) @ matrix_cell @ self.matrix_interface(med_exit, meds[-1])
        else:
            temp = la.matrix_power(self.matrix_interface(meds[0],meds[-1]) @ matrix_cell, self.p -1)
            return self.matrix_interface(meds[0], med_entry) @ matrix_cell @ temp @ self.matrix_interface(med_exit, meds[-1])
    
    def plot_field(self, func, save=None):
        self.calc_complex_amplitudes()
        fig, ax = plt.subplots(dpi=240)
        fig.set_size_inches(6,4)
        self.plot_layers(ax)
        ax.set_title("Spatial field distribution", fontfamily="Arial", fontsize=14)
        z_struct = 0
        if self.mode == 'acoustic':
            ks = [2*np.pi/(self.wl * mat.c / self.entry.c) for mat in self.maters]
            co = '#1a1a00'
        elif self.mode == 'optical' :
            ks = [2*np.pi*mat.n/self.wl for mat in self.maters]
            co = '#00134d'
        l = len(self.maters)
        for i in range(1,self.p + 1):
            for j in range(l):
                eplus, eminus = (self.Es[l*(i-1)+1+j][0],self.Es[l*(i-1)+1+j][1])
                zs = np.linspace(z_struct,z_struct + self.ds[j],50)
                ax.plot(zs,[func(eplus, eminus, ks[j], z-z_struct) for z in zs],color=co,lw=1.2)
                z_struct += self.ds[j]
        if save is not None:
            plt.savefig(save+".png")
        plt.show()
    
    def plot_layers(self, ax):
        z = 0
        for i in range(self.p):
            for i in range(len(self.maters)):
                ax.add_patch(Rectangle((z, 0), self.ds[i], 1,
             edgecolor = 'black',
             facecolor = self.maters[i].color,
             fill=True,
             alpha = 0.5,
             lw=1))
                z += self.ds[i]
        ax.set_xlim(-sum(self.ds),sum(self.ds)*(self.p + 1))
        ax.set_ylim(-0.1,1.1)
        ax.set_xlabel("Position (nm)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        if self.mode == 'acoustic':
            ax.set_ylabel("Normalized acoustic field |u|",fontsize=8)
        elif self.mode == 'optical':
            ax.set_ylabel("Normalized electric field |E|",fontsize=8)
        
                
                
            
    def calc_complex_amplitudes(self,noreverse=False, zerostress=False):
        if self.mode == 'acoustic':
            ks = [2*np.pi/(self.wl * mat.c / self.entry.c) for mat in self.maters]
            meds = [mat.Z for mat in self.maters]                   
            med_entry, med_exit = self.entry.Z, self.exit.Z
        elif self.mode == 'optical' :
            ks = [2*np.pi*mat.n/self.wl for mat in self.maters]
            meds = [mat.n for mat in self.maters]
            med_entry, med_exit = self.entry.n, self.exit.n
        if zerostress:
            self.Es = [[1+0*1j,1+0*1j]]
            self.Es.append(list(self.matrix_prop(ks[-1],-self.ds[-1]) @ self.Es[-1]))
            for i in range(len(self.maters)-2,-1,-1):
                self.Es.append(list(self.matrix_prop(ks[i],-self.ds[i]) @ self.matrix_interface(meds[i+1], meds[i]) @ self.Es[-1]))
        else:
            self.Es.append(list(self.matrix_prop(ks[-1],-self.ds[-1]) @ self.matrix_interface(med_exit, meds[-1]) @ self.Es[-1]))
            for i in range(len(self.maters)-2,-1,-1):
                self.Es.append(list(self.matrix_prop(ks[i],-self.ds[i]) @ self.matrix_interface(meds[i+1], meds[i]) @ self.Es[-1]))
        for _ in range(self.p - 1):
            self.Es.append(list(self.matrix_prop(ks[-1],-self.ds[-1]) @ self.matrix_interface(meds[0], meds[-1]) @ self.Es[-1]))
            for i in range(len(self.maters)-2,-1,-1):
                self.Es.append(list(self.matrix_prop(ks[i],-self.ds[i]) @ self.matrix_interface(meds[i+1], meds[i]) @ self.Es[-1]))
        self.Es.append(list(self.matrix_interface(meds[0],med_entry) @ self.Es[-1]))
        if noreverse:
            return
        self.Es.reverse()
            
    def field_value(self, Eplus, Eminus, k_med, z):
        return np.real(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))#/(abs(self.Es[0][0]))
    
    def field_value2(self, Eplus, Eminus, k_med, z):
        return np.abs(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(2*abs(self.Es[0][0]))
    
    def calc_refl_trans(self):
        M = self.build_transfer_matrix()
        a,b = M @ np.array([1,0])
        if self.mode == 'optical':
            return (abs(b/a)**2, abs(1/a)**2*self.exit.n/self.entry.n)
        else:
            return (abs(b/a)**2, abs(1/a)**2*self.exit.Z/self.entry.Z)
    
    def Es_initial(self, e0):
        self.Es = e0
    
    def photoelastic_index(self):
        temp = np.empty(len(self.maters),dtype='bool')
        for i in range(len(self.maters)):
            if self.maters[i].name == "GaAs":
                temp[i] = True
            else:
                temp[i] = False
        return np.tile(temp,self.p).nonzero()[0] + 1
        
