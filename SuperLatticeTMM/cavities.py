
"""
Spyder Editor
Fabry Perot Resonators and Cavities
v 1.01
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from SuperlatticeTMM.superlattice import Superlattice
from math import ceil
from matplotlib.lines import Line2D
from numpy import linalg as LA

class OpticalCavity:
    def __init__(self, maters, ds, wl, entry, exit, p, spacer):
        self.maters = maters
        self.ds = ds
        self.p = p
        self.wl = wl
        self.entry = entry
        self.spacer = spacer
        self.exit = exit
        self.Es = []
        self.ks = [2*np.pi*mat.n/self.wl for mat in self.maters]
        self.kspacer= 2*np.pi*self.spacer[0].n/self.wl 
        self.sname = r"${%s}|DBR\left ( \lambda/4,\lambda/4 \right )_{%d} |Spacer(\lambda)|DBR\left (\lambda/4,\lambda/4 \right )_{%d}|{%s}$ @ %d nm" % (self.entry.name, self.p[0], self.p[1], self.exit.name,self.wl)
        
    def build_TM(self):
        dbr1 = Superlattice(self.maters[::-1],self.ds[::-1],self.wl,self.spacer[0], self.exit, p=self.p[1], mode='optical').build_transfer_matrix()
        spacer_m = self.matrix_prop(self.kspacer, -self.spacer[1])
        dbr2 = Superlattice(self.maters,self.ds,self.wl,self.entry, self.spacer[0], p=self.p[0],mode='optical').build_transfer_matrix()
        return dbr2 @ spacer_m @ dbr1
    
    def field_value(self, Eplus, Eminus, k_med, z):
        return np.real(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(abs(self.Es[0][0]))
    
    def field_value2(self, Eplus, Eminus, k_med, z):
        return np.abs(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(abs(self.Es[0][0]))
    
    def matrix_interface(self, ni, nj):
        return np.array([[1 + ni/nj, 1 - ni/nj], [1 - ni/nj, 1 + ni/nj]]) * 0.5
   
    def matrix_prop(self, k_med, d):
        phase = 1j*k_med*d
        return np.array([[np.exp(phase),0],[0,np.exp(-phase)]])
    
    def calc_complex_amplitudes(self):
        dbr = Superlattice(self.maters[::-1],self.ds[::-1],self.wl,self.spacer[0], self.exit, p=self.p[1], mode='optical')
        dbr.calc_complex_amplitudes(noreverse=True)
        self.Es.extend(dbr.Es)
        self.Es[-1] = list(self.matrix_prop(self.kspacer,-self.spacer[1]) @ self.Es[-1])
        dbr = Superlattice(self.maters,self.ds,self.wl,self.entry, self.spacer[0], p=self.p[0],mode='optical')
        dbr.Es_initial([self.Es[-1]])
        dbr.calc_complex_amplitudes(noreverse=True)
        self.Es.extend(dbr.Es[1:])
        self.Es.reverse()
        
    def plot_field(self, plot_func, save=None, axe=None, h=0):
        self.calc_complex_amplitudes()
        if axe is None:
            fig, axe = plt.subplots(dpi=300)
            fig.set_size_inches(6,3)
        axe.set_xlabel("Position (nm)", fontsize=11)
        axe.tick_params(axis='both', which='major', labelsize=8)
        if plot_func == 'real':
            axe.set_ylabel("Re(E)(Norm.)")
            func = self.field_value
        elif plot_func == 'modulus':
            axe.set_ylabel("Normalized electric field |E|",fontsize=8)
            func = self.field_value2
        self.plot_layers(axe,h)
        z_struct = 0
        l = len(self.maters)
        for i in range(1,self.p[0] + 1):
            for j in range(l):
                eplus, eminus = (self.Es[l*(i-1)+1+j][0],self.Es[l*(i-1)+1+j][1])
                zs = np.linspace(z_struct,z_struct + self.ds[j],100*ceil(abs(eplus)))
                axe.plot(zs,[func(eplus, eminus, self.ks[j], z-z_struct) for z in zs],color="#E6291B",lw=1.5)
                z_struct += self.ds[j]
        eplus, eminus = (self.Es[l*self.p[0]+1][0],self.Es[l*self.p[0]+1][1])
        zs = np.linspace(z_struct,z_struct + self.spacer[1],100*ceil(abs(eplus)))
        axe.plot(zs,[func(eplus, eminus, self.kspacer, z-z_struct) for z in zs],color="#E6291B",lw=1.5)
        z_struct += self.spacer[1]
        for i in range(self.p[1]):
            for j in range(l):
                offset = l*self.p[0]+1
                eplus, eminus = (self.Es[l*i+1+j+offset][0],self.Es[l*i+1+j+offset][1])
                zs = np.linspace(z_struct,z_struct + self.ds[-1-j],100*ceil(abs(eplus)))
                axe.plot(zs,[func(eplus, eminus, self.ks[-1-j], z-z_struct) for z in zs],color="#E6291B",lw=1.5)
                z_struct += self.ds[-1-j]

        custom_lines = [Line2D([0], [0], color='#aea6f7', lw=2),
                Line2D([0], [0], color='#91cbfa', lw=2)]

        axe.legend(custom_lines, ['GaAs', 'AlAs'],loc='upper left')
        if save is not None:
            plt.savefig(save+".png")
        
    
    
    def plot_layers(self,ax, h):
        z = 0
        if h == 0:
            h = abs(self.Es[2*self.p[0]+1][0] + self.Es[2*self.p[1]+1][1]) / abs(self.Es[0][0])
        for i in range(self.p[0]):
            for j in range(len(self.maters)):
                ax.add_patch(Rectangle((z, -1.05*h), self.ds[j], 2.1*h,
                 edgecolor = 'black',
                 facecolor = self.maters[j].color,
                 fill=True,
                 alpha = 0.3,
                 lw=1))
                z += self.ds[j]
        ax.add_patch(Rectangle((z, -1.05*h), self.spacer[1], 2.1*h,
             edgecolor = 'black',
             facecolor = self.spacer[0].color,
             fill=True,
             alpha = 0.3,
             lw=1))
        z += self.spacer[1]
        for _ in range(self.p[1]):
            for j in range(len(self.maters)):
                ax.add_patch(Rectangle((z, -1.05*h), self.ds[-1-j], 2.1*h,
                 edgecolor = 'black',
                 facecolor = self.maters[-1-j].color,
                 fill=True,
                 alpha = 0.3,
                 lw=1))
                z += self.ds[-1-j]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text((self.ds[0] + self.ds[1])*(self.p[0] + self.p[1] -3),-0.8*h,f"λ= {self.wl} nm",bbox=props)
        ax.set_xlim(-sum(self.ds),sum(self.ds)*(self.p[0] + self.p[1] + 1) + self.spacer[1])
        ax.set_ylim(-1.05*h,1.05*h)
    
    
    def calc_refl_trans(self):
        M = self.build_TM()
        a,b = M @ np.array([1,0])
        if self.mode == 'optical':
            return (abs(b/a)**2, abs(1/a)**2*self.exit.n/self.entry.n)
        else:
            return (abs(b/a)**2, abs(1/a)**2*self.exit.Z/self.entry.Z)
    
    def calc_spectra(self, wls):
        refls = []
        temp = self.wl
        temp2 = self.kspacer
        for wla in wls:
            self.wl = wla
            self.kspacer= 2*np.pi*self.spacer[0].n/self.wl
            M = self.build_TM()
            a,b = M @ np.array([1,0])
            refls.append(abs(b/a)**2)
        self.wl = temp
        self.kspacer = temp2
        return refls
    
    #def plot_summary(self, wls, wl_DBR,save=None):
        #fig, axes = plt.subplots(2, dpi=300)
        #fig.set_size_inches(10,8)
        #fig.suptitle(self.sname)
        #axes[0].plot(wls,self.calc_spectra(wls))
        #axes[0].set_ylabel("Opt.Reflectivity")
        #axes[0].set_xlabel("Wavelength(nm)")
        #self.plot_field('real',axe=axes[1])
        #axes[1].tick_params(axis='x', which='major', labelsize=10)
        #wl_zoom = np.linspace(wl_DBR-3,wl_DBR+3,200)
        #axes[2].plot(wl_zoom, self.calc_spectra(wl_zoom))
        #axes[2].set_ylabel("Opt.Reflectivity")
        #axes[2].set_xlabel("Wavelength(nm)")
        #fig.tight_layout(pad=2.0)
        #if save is not None:
            #plt.savefig(save + ".png",dpi=300)
    
    def photoelastic_index(self):
        temp = []
        for i in range(len(self.maters)):
            if self.maters[i].name == "GaAs":
                temp.append(True)
            else:
                temp.append(False)
        return np.array(temp*self.p[0] + [self.spacer[0].name == "GaAs"] + (temp*self.p[1])[::-1]).nonzero()[0] + 1
            
    
    

    
    
class Resonator:
    def __init__(self, maters, ds, wl, entry, exit, p, spacer):
        self.maters = maters
        self.ds = ds
        self.p = p
        self.wl = wl
        self.entry = entry
        self.spacer = spacer
        self.exit = exit
        #self.mode = mode
        self.Es = []
        self.ks = [2*np.pi/(self.wl * mat.c / self.entry.c) for mat in self.maters]
        self.kspacer = 2*np.pi/(self.wl*self.spacer[0].c/self.entry.c)
        self.sname = r"${%s}|\times %d \left ( \lambda/4,\lambda/4 \right ) |Spacer(\lambda)|\times %d\left (\lambda/4,\lambda/4 \right )|{%s}$" % (self.entry.name, self.p[0], self.p[1], self.exit.name) + "\n @ 870 nm"
        
    def build_TM(self):
        dbr1 = Superlattice(self.maters[::-1],self.ds[::-1],self.wl,self.spacer[0], self.exit, p=self.p[1], mode='acoustic').build_transfer_matrix()
        spacer_m = self.matrix_prop(self.kspacer, -self.spacer[1])
        dbr2 = Superlattice(self.maters,self.ds,self.wl,self.entry, self.spacer[0], p=self.p[0],mode='acoustic').build_transfer_matrix()
        return dbr2 @ spacer_m @ dbr1
    
    def field_value(self, Eplus, Eminus, k_med, z):
        return np.real(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(abs(self.Es[0][0]))
    
    def field_value2(self, Eplus, Eminus, k_med, z):
        return np.abs(Eplus*np.exp(1j*k_med*z)+Eminus*np.exp(-1j*k_med*z))/(abs(self.Es[0][0]))
    
    def matrix_interface(self, ni, nj):
        return np.array([[1 + ni/nj, 1 - ni/nj], [1 - ni/nj, 1 + ni/nj]]) * 0.5
   
    def matrix_prop(self, k_med, d):
        phase = 1j*k_med*d
        return np.array([[np.exp(phase),0],[0,np.exp(-phase)]])
    
    def calc_complex_amplitudes(self):
        dbr = Superlattice(self.maters[::-1],self.ds[::-1],self.wl,self.spacer[0], self.exit, p=self.p[1], mode='acoustic')
        #dbr.Es_initial([[1.,1.]])
        #dbr.calc_complex_amplitudes(noreverse=True)
        dbr.calc_complex_amplitudes(noreverse=True, zerostress=True)
        self.Es.extend(dbr.Es)
        self.Es[-1] = list(self.matrix_prop(self.kspacer,-self.spacer[1]) @ self.Es[-1])
        dbr = Superlattice(self.maters,self.ds,self.wl,self.entry, self.spacer[0], p=self.p[0],mode='acoustic')
        dbr.Es_initial([self.Es[-1]])
        dbr.calc_complex_amplitudes(noreverse=True)
        self.Es.extend(dbr.Es[1:])
        self.Es.reverse()
        
    def plot_field(self, plot_func, save=None,h=0, axe=None):
        self.calc_complex_amplitudes()
        if axe is None:
            fig, axe = plt.subplots(dpi=300)
            fig.set_size_inches(6,3)
        axe.set_xlabel("Position (nm)", fontsize=8)
        axe.tick_params(axis='both', which='major', labelsize=8)
        if plot_func == 'real':
            axe.set_ylabel("Displacement (norm.)",fontsize=8)
            func = self.field_value
        elif plot_func == 'modulus':
            axe.set_ylabel("Normalized acoustic field |u|",fontsize=8)
            func = self.field_value2
        self.plot_layers(axe,h)
        z_struct = 0
        l = len(self.maters)
        for i in range(self.p[0]):
            for j in range(l):
                eplus, eminus = (self.Es[i*l+1+j][0], self.Es[i*l+1+j][1])
                zs = np.linspace(z_struct,z_struct + self.ds[j],80*ceil(abs(eplus)))
                axe.plot(zs,[func(eplus, eminus, self.ks[j], z-z_struct) for z in zs],color="#07357d",lw=1)
                z_struct += self.ds[j]
                
        eplus, eminus = (self.Es[l*self.p[0]+1][0],self.Es[l*self.p[0]+1][1])
        zs = np.linspace(z_struct,z_struct + self.spacer[1],80*ceil(abs(eplus)))
        axe.plot(zs,[func(eplus, eminus, self.kspacer, z-z_struct) for z in zs],color="#07357d",lw=1)
        z_struct += self.spacer[1]
        offset = l*self.p[0] + 1
        for i in range(self.p[1]):
            for j in range(l):
                eplus, eminus = (self.Es[i*l+1+j+offset][0], self.Es[i*l+1+j+offset][1])
                zs = np.linspace(z_struct,z_struct + self.ds[-1-j],80*ceil(abs(eplus)))
                axe.plot(zs,[func(eplus, eminus, self.ks[-1-j], z-z_struct) for z in zs],color="#07357d",lw=1)
                z_struct += self.ds[-1-j]
        custom_lines = [Line2D([0], [0], color='#aea6f7', lw=2),
                Line2D([0], [0], color='#91cbfa', lw=2)]

        axe.legend(custom_lines, ['GaAs', 'AlAs'],loc='upper left')
        if save is not None:
            plt.savefig(save+".png")
        
    
    
    def plot_layers(self,ax,h):
        z = 0
        if h == 0:
            h = abs(self.Es[2*self.p[0]+1][0] + self.Es[2*self.p[0]+1][1]) / abs(self.Es[0][0])
        for i in range(self.p[0]):
            for j in range(len(self.maters)):
                ax.add_patch(Rectangle((z, -1.05*h), self.ds[j], 2.1*h,
                 edgecolor = 'black',
                 facecolor = self.maters[j].color,
                 fill=True,
                 alpha = 0.3,
                 lw=1))
                z += self.ds[j]
        ax.add_patch(Rectangle((z, -1.05*h), self.spacer[1], 2.1*h,
             edgecolor = 'black',
             facecolor = self.spacer[0].color,
             fill=True,
             alpha = 0.3,
             lw=1))
        z += self.spacer[1]
        for _ in range(self.p[1]):
            for j in range(len(self.maters)):
                ax.add_patch(Rectangle((z, -1.05*h), self.ds[-1-j], 2.1*h,
                 edgecolor = 'black',
                 facecolor = self.maters[-1-j].color,
                 fill=True,
                 alpha = 0.3,
                 lw=1))
                z += self.ds[-1-j]
        ax.set_xlim(-sum(self.ds),sum(self.ds)*(self.p[0] + self.p[1] + 1) + self.spacer[1])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text((self.ds[0] + self.ds[1])*(self.p[0] + self.p[1] -3),-0.8*h,f"f= {4780/self.wl:.2f} GHz", bbox=props)
        ax.set_ylim(-1.05*h,1.05*h)
    
    
    def calc_surface_disp(self,spacer=False):
        if len(self.Es) < 1:
            self.calc_complex_amplitudes()
        if spacer:
            i = len(self.maters)*self.p[0]+1
            return (abs(self.Es[i][0]))**2
        return (LA.norm(np.asarray(self.Es[-1]))/LA.norm(np.asarray(self.Es[0])))**2
    
    def photoelastic_index(self):
        temp = []
        for i in range(len(self.maters)):
            if self.maters[i].name == "GaAs":
                temp.append(True)
            else:
                temp.append(False)
        return np.array(temp*self.p[0] + [self.spacer[0].name == "GaAs"] + (temp*self.p[1])[::-1]).nonzero()[0] + 1
    
    def calc_spectra(self, wls):
        refls = []
        temp = self.wl
        temp2 = self.kspacer
        for wla in wls:
            self.wl = wla
            self.kspacer = 2*np.pi/(self.wl*self.spacer[0].c/self.entry.c)
            M = self.build_TM()
            a,b = M @ np.array([1,0])
            refls.append(abs(b/a)**2)
        self.wl = temp
        self.kspacer = temp2
        return refls
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
        
