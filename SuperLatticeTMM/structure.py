# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import materials
from math import ceil
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

class StructureOA:
    
    def __init__(self, params, alloys=(0,0)):
        self.maters = params['materials']
        self.ds = params['thickness']
        if alloys == (0,0):    
            self.matdict = {"GaAs":materials.GaAs, "AlAs":materials.AlAs, "Air":materials.Air}
        else:
            self.matdict = {"GaAs":materials.AlGaAsx(alloys[0]), "AlAs":materials.AlGaAsx(alloys[1]), "Air":materials.Air}
        self.U_vectors = np.empty((self.ds.size+2,2), dtype='complex')
        self.E_vectors = np.empty((self.ds.size+2,2), dtype='complex')
        
    def matrix_interface(self, ni, nj):
        return np.array([[1 + ni/nj, 1 - ni/nj], [1 - ni/nj, 1 + ni/nj]]) * 0.5
    
    def matrix_prop(self, k_med, d):
        phase = 1j*k_med*d
        return np.array([[np.exp(phase),0],[0,np.exp(-phase)]])
    
    def field_value(self, Eplus, Eminus, k_med, z):
        return np.real(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))#/(abs(self.Es[0][0]))
   
    def field_value2(self, Eplus, Eminus, k_med, z):
        return np.abs(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))#/(abs(self.Es[0][0]))
    
    
    def build_TM(self, wl, mode, entry, ex):
        if mode =='optic':
            ks = [2*np.pi*self.matdict[mat].n/wl for mat in self.maters]
            meds = [self.matdict[mat].n for mat in self.maters]
            med_entry, med_exit = self.matdict[entry].n, self.matdict[ex].n
        elif mode == 'acoustic':
            ks = [2*np.pi/(wl * self.matdict[mat].c / self.matdict[entry].c) for mat in self.maters]
            meds = [self.matdict[mat].Z for mat in self.maters]                   
            med_entry, med_exit = self.matdict[entry].Z, self.matdict[ex].Z
        tm = np.identity(2, dtype='complex')
        for i in range(self.ds.size-1,0,-1):
            tm =  self.matrix_interface(meds[i], meds[i-1]) @ self.matrix_prop(ks[i], -self.ds[i]) @ tm
        tm = self.matrix_prop(ks[0], -self.ds[0]) @ tm
        return self.matrix_interface(meds[0], med_entry) @ tm  @ self.matrix_interface(med_exit, meds[-1])
    
    def get_optic_refl(self, wl_range, entry, ex):
        refs = np.empty(wl_range.shape)
        it = np.nditer(wl_range, flags=['f_index'])
        for wla in it:
            a, b = self.build_TM(wla,'optic',entry, ex) @ np.array([1,0])
            refs[it.index] = abs(b/a)**2
        return refs
    
    def get_acoustic_refl(self, freq_range, entry, ex):
        refs = np.empty(freq_range.shape)
        it = np.nditer(freq_range, flags=['f_index'])
        for freq in it:
            a, b = self.build_TM(self.matdict[entry].c/freq,'acoustic',entry, ex) @ np.array([1,0])
            refs[it.index] = abs(b/a)**2
        return refs
    
    def calc_complex_Evectors(self, wl, entry, ex):
        ks = [2*np.pi*self.matdict[mat].n/wl for mat in self.maters]
        meds = [self.matdict[mat].n for mat in self.maters]
        med_entry, med_exit = self.matdict[entry].n, self.matdict[ex].n
        self.E_vectors[-1] = np.array([1,0],dtype='complex')
        self.E_vectors[-2] = (self.matrix_prop(ks[-1],-self.ds[-1]) @ self.matrix_interface(med_exit,meds[-1]) @ self.E_vectors[-1])
        for i in range(self.ds.size-2,-1,-1):
            self.E_vectors[i+1] = self.matrix_prop(ks[i],-self.ds[i]) @ self.matrix_interface(meds[i+1],meds[i]) @ self.E_vectors[i+2]
        self.E_vectors[0] = self.matrix_interface(meds[0],med_entry) @ self.E_vectors[1]
    
    def calc_complex_Uvectors(self, freq, entry, ex, zerostrain=True):
        wl = self.matdict[entry].c / freq
        ks = [2*np.pi/(wl * self.matdict[mat].c / self.matdict[entry].c) for mat in self.maters]
        meds = [self.matdict[mat].Z for mat in self.maters]                   
        med_entry, med_exit = self.matdict[entry].Z, self.matdict[ex].Z
        if not zerostrain: 
            self.U_vectors[-1] = np.array([1,0],dtype='complex')
            self.U_vectors[-2] = (self.matrix_prop(ks[-1],-self.ds[-1]) @ self.matrix_interface(med_exit,meds[-1]) @ self.U_vectors[-1])
        else:
            self.U_vectors[-1] = np.array([1,1], dtype='complex')
            self.U_vectors[-2] = self.matrix_prop(ks[-1],-self.ds[-1]) @ self.U_vectors[-1]
        for i in range(self.ds.size-2,-1,-1):
            self.U_vectors[i+1] = self.matrix_prop(ks[i],-self.ds[i]) @ self.matrix_interface(meds[i+1],meds[i]) @ self.U_vectors[i+2]
        self.U_vectors[0] = self.matrix_interface(meds[0],med_entry) @ self.U_vectors[1]
    
    def plot_field(self, field_type, plot_func, f_or_wl, entry, exit, save=None, axe=None, h=5):
        if axe is None:
            fig, axe = plt.subplots(dpi=300)
            fig.set_size_inches(6,3)
        axe.set_xlabel("Position (nm)", fontsize=11)
        axe.tick_params(axis='both', which='major', labelsize=8)
        if field_type == 'optic':
            ks = [2*np.pi*self.matdict[mat].n/f_or_wl for mat in self.maters]
            col="#E6291B"
            self.calc_complex_Evectors(f_or_wl, entry, exit)
            vects = self.E_vectors
            if plot_func == 'real':
                axe.set_ylabel("Re(E)(norm.)")
                func = self.field_value
            elif plot_func == 'modulus':
                axe.set_ylabel("Electric field amplitude (norm.)",fontsize=8)
                func = self.field_value2
        else:
            col = "#07357d"
            self.calc_complex_Uvectors(f_or_wl, entry, exit)
            ks = [2*np.pi/(self.matdict[mat].c/f_or_wl) for mat in self.maters]
            vects = self.U_vectors
            if plot_func == 'real':
                axe.set_ylabel("Displacement (norm.)",fontsize=8)
                func = self.field_value
            elif plot_func == 'modulus':
                axe.set_ylabel("Displacement amplitude (norm.)",fontsize=8)
                func = self.field_value2
        self.plot_layers(axe,h)
        z_struct = 0
        for i in range(1,self.ds.size + 1):
            eplus, eminus = vects[i][0], vects[i][1]
            zs = np.linspace(z_struct,z_struct + self.ds[i-1],100*ceil(abs(eplus)))
            axe.plot(zs,func(eplus, eminus, ks[i-1], zs-z_struct)/abs(vects[0][0]),color=col,lw=1.2)
            z_struct += self.ds[i-1]
        custom_lines = [Line2D([0], [0], color=self.matdict["GaAs"].color, lw=2),
                Line2D([0], [0], color=self.matdict["AlAs"].color, lw=2)]
        axe.legend(custom_lines, ['GaAs', 'AlAs'],loc='upper left')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        th_struct = np.sum(self.ds)
        if plot_func == 'real':
            axe.set_ylim(-1.05*h,1.05*h)
            if field_type == 'optic':
                axe.text(0.8*th_struct,-0.8*h,f"λ= {f_or_wl} nm",bbox=props)
            else:
                axe.text(0.8*th_struct,-0.8*h,f"f= {f_or_wl:.2f} GHz", bbox=props)
            
        else:
            axe.set_ylim(-0.08*h,1.05*h)
            if field_type == 'optic':
                axe.text(0.8*th_struct,0.5*h,f"λ= {f_or_wl} nm",bbox=props)
            else:
                axe.text(0.8*th_struct,0.5*h,f"f= {f_or_wl:.2f} GHz", bbox=props)
        axe.set_xlim(-100, th_struct + 100)
            
        if save is not None:
            plt.savefig(save+".png")
        
        
    def plot_layers(self,ax, h):
        z = 0
        for i in range(self.ds.size):
            ax.add_patch(Rectangle((z, -1.05*h), self.ds[i], 2.1*h,
                 edgecolor = 'black',
                 facecolor = self.matdict[self.maters[i]].color,
                 fill=True,
                 alpha = 0.3,
                 lw=1))
            z += self.ds[i]