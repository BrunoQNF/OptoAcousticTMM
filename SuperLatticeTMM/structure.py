# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import materials

class StructureOA:
    
    def __init__(self, params, alloys=(0,0)):
        self.ds = [x[1] for x in params]
        self.maters = [x[0] for x in params]
        if alloys == (0,0):    
            self.matdict = {"GaAs":materials.GaAs, "AlAs":materials.AlAs, "Air":materials.Air}
        else:
            self.matdict = {"GaAs":materials.AlGaAsx(alloys[0]), "AlAs":materials.AlGaAsx(alloys[1]), "Air":materials.Air}
        
    def matrix_interface(self, ni, nj):
        return np.array([[1 + ni/nj, 1 - ni/nj], [1 - ni/nj, 1 + ni/nj]]) * 0.5
    
    def matrix_prop(self, k_med, d):
        phase = 1j*k_med*d
        return np.array([[np.exp(phase),0],[0,np.exp(-phase)]])
    
    def field_value(self, Eplus, Eminus, k_med, z):
        return np.real(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(abs(self.Es[0][0]))
   
    def field_value2(self, Eplus, Eminus, k_med, z):
        return np.abs(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(abs(self.Es[0][0]))
    
    
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
        for i in range(len(self.ds)-1,0,-1):
            tm =  self.matrix_interface(meds[i], meds[i-1]) @ self.matrix_prop(ks[i], -self.ds[i]) @ tm
        tm = self.matrix_prop(ks[0], -self.ds[0]) @ tm
        return self.matrix_interface(meds[0], med_entry) @ tm  @ self.matrix_interface(med_exit, meds[-1])
    
    def get_optic_refl(self, wl_range, entry, ex):
        refs = np.empty(wl_range.shape)
        it = np.nditer(wl_range, flags=['c_index'])
        for wla in it:
            a, b = self.build_TM(wla,'optic',entry, ex) @ np.array([1,0])
            refs[it.index] = abs(b/a)**2
        return refs
    
    def get_acoustic_refl(self, freq_range, entry, ex):
        refs = np.empty(freq_range.shape)
        it = np.nditer(freq_range, flags=['c_index'])
        for freq in it:
            a, b = self.build_TM(self.matdict[entry].c/freq,'acoustic',entry, ex) @ np.array([1,0])
            refs[it.index] = abs(b/a)**2
        return refs