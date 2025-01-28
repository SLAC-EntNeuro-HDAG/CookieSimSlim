#!/usr/bin/python3

import sys
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2
import multiprocessing as mp
from pyJoules.energy_meter import measure_energy

from utils import gauss,addGauss2d,addGauss2d_padding_10,detect_peaks

#NSHOTS = 1<<3
#NSHOTS = 1<<6 # roughly 2 seconds per shot, so 1<<6 = 2.5 min
NSHOTS = 1<<8 # roughly 10 minutes
#NSHOTS = 1<<10 # roughly 40 minutes
#NSHOTS = 1<<12 # 
TEPROD = 1.62
TWIN=4./.3 #4 micron streaking/(.3microns/fs)
EWIN=100. #100 eV window
THRESH = 15


def kernConv1d(kmat,xmat):
    res = np.zeros(xmat.shape[1],dtype=float)
    nprocs = min(mp.cpu_count(),xmat.shape[0])
    ind = 0
    rmax = 0
    vmax = 0.
    for i in range(xmat.shape[0]):
        tmp = np.sum( np.fft.ifft(np.fft.fft(np.roll(xmat,i,axis=0),axis=1)*np.fft.fft(np.flip(kmat,axis=1),axis=1),axis=1).real, axis=0)
        if np.max(tmp)>np.max(res):
            res = tmp
            ind = i
            rmax = np.argmax(res)
            vmax = res[rmax]
    return ind,rmax,vmax,res

def fftconv(X):
    tmp = np.sum( np.fft.ifft(np.fft.fft(np.roll(X['xmat'],X['i'],axis=0),axis=1)*np.fft.fft(np.flip(X['kmat'],axis=1),axis=1),axis=1).real, axis=0)
    return (X['i'],tmp)

def para_kernConv1d(kmat,xmat):
    res = np.zeros(xmat.shape[1],dtype=float)
    nprocs = min(mp.cpu_count(),xmat.shape[0])
    ind = 0
    rmax = 0
    vmax = 0.
    Xlist = []
    for i in range(xmat.shape[0]):
        Xlist += [{'i':i,'kmat':kmat,'xmat':xmat}]

    with mp.Pool(nprocs) as p:
        tmplist = p.map(fftconv,Xlist)
        for i in range(len(tmplist)):
            if np.max(tmplist[i][1])>np.max(res):
                res = tmplist[i][1]
                ind = tmplist[i][0]
                rmax = np.argmax(res)
                vmax = res[rmax]
    return ind,rmax,vmax,res

def conv1d(xmat,ymat):
    return np.fft.ifft(np.fft.fft(xmat,axis=1)*np.fft.fft(np.flip(ymat,axis=1),axis=1),axis=1)

def scanKernel(widths,strengths,xmat):
    kmat = np.zeros(xmat.shape,dtype=float)
    vref = 0.
    vmax = 0.
    stref = 0.
    wdref = 0.
    indref = 0
    rowref = 0
    if not np.max(xmat)>0:
        return indref,rowref,stref,wdref,vref
    for st in strengths:
        for wd in widths:
            kmat = fillKernel(wd,st,kmat)
            ind,row,vmax,res = kernConv1d(kmat,xmat)
            if vmax>vref:
                indref = ind
                rowref = row
                wdref = wd
                stref = st
                vref = vmax
    return indref,rowref,stref,wdref,vref

def fillKernel(width,strength,kern):
    for r in range(kern.shape[0]):
        w = width 
        c = float(kern.shape[1]>>1) + strength * np.sin(r*2*np.pi/float(kern.shape[0]))
        kern[r,:] = gauss(np.arange(kern.shape[1]),w,c)
    #norm = math.sqrt(np.sum(kern))
    norm = math.sqrt(np.inner(kern.flatten(),kern.flatten()))
    return kern/norm

@measure_energy
def main(fname,plotting=False):
    rng = np.random.default_rng()
    print('For now assuming 4 micron wavelength streading for a 13 fs temporal window')
    print('For now also assuming 100eV for the energy window regardless of sampling')
    print('.1eV = 100fs, 1eV = 10fs, (30nm width @ 800nm) 50meV = 33fs gives time-bandwidth product dE[eV]*dt[fs] = %f'%TEPROD)
    runtimes = []
    with h5py.File(fname,'r') as f:
        shotkeys = [k for k in f.keys() if len(f[k].attrs['sasecenters'])>0]
        rng.shuffle(shotkeys)
        oname = fname + '.tempout.h5'
        m = re.search('^(.*/)(\w+.+\.\d+)\.h5',fname)
        if m:
            oname = m.group(2) + 'confusion.h5'
            opath = m.group(1) + 'output/'
            if not os.path.exists(opath):
                os.mkdir(opath)
        else:
            print('using default output file in current directory')

        temat = np.zeros(f[shotkeys[0]]['Ximg'].shape,dtype=float)
        tedist = np.zeros((f[shotkeys[0]]['Ximg'].shape[0]+10,f[shotkeys[0]]['Ximg'].shape[1]+10),dtype=float)

        with h5py.File(opath + '/' + oname,'w') as o:
            if 'shotkeys' in o.keys():
                o['true'].attrs['hist'] = np.zeros((1<<4),dtype=np.uint16)
                o['pred']=np.zeros((1<<4,1<<4),dtype=np.uint16)
                o['pred'].attrs['true']=0
                o['pred'].attrs['hist'] = np.zeros((1<<4),dtype=np.uint16)
                o['shotkeys']=shotkeys[:NSHOTS]
                o['coeffhist']=np.zeros((1<<7),dtype=np.uint32)
                o['coeffbins']=np.arange((1<<7)+1,dtype=float)/float(1<<7)
            else: 
                o.create_group('true')
                o['true'].attrs.create('hist',data = np.zeros((1<<4),dtype=np.uint16))
                o.create_dataset('pred',data = np.zeros((1<<4,1<<4),dtype=np.uint16))
                o['pred'].attrs.create('true',data=0)
                o['pred'].attrs.create('hist',data = np.zeros((1<<4),dtype=np.uint16))
                o.create_dataset('shotkeys',data = shotkeys[:NSHOTS])
                o.create_dataset('coeffhist',data = np.zeros(1<<7,dtype=np.uint32))
                o.create_dataset('coeffbins',data = np.arange((1<<7)+1,dtype=float)/float(1<<7))

            coefflist = []
            nsase={'true':0}
            nsase={'pred':0}

            for k in shotkeys[:NSHOTS]:
                temat = np.zeros(f[shotkeys[0]]['Ximg'].shape,dtype=float)
                tedist = np.zeros((f[shotkeys[0]]['Ximg'].shape[0]+20,f[shotkeys[0]]['Ximg'].shape[1]+20),dtype=float)
                tepeaks = np.zeros((f[shotkeys[0]]['Ximg'].shape[0]+20,f[shotkeys[0]]['Ximg'].shape[1]+20),dtype=float)

                t0 = time.time()

                nsase={'true':0}

                nsase['true'] = f[k].attrs['sasecenters'].shape[0]
                if nsase['true']>10:
                    print('skipping nsase = %i'%(nsase['true']))
                    continue

                x = np.copy(f[k]['Ximg'][()]).astype(int) # deep copy to preserve original
                y = np.copy(f[k]['Ypdf'][()]).astype(float) # deep copy to preserve original
                tstep = TWIN/x.shape[0]
                estep = EWIN/x.shape[1]
                wlist = f[k].attrs['sasewidth']*estep*np.arange(.25,1.75,.125,dtype=float)
                #print(wlist)
                slist = f[k].attrs['kickstrength']*np.arange(.25,2,.125,dtype=float)
                #print(slist)
                kern = np.zeros(x.shape,dtype=float)
                proj_display = np.zeros(x.shape,dtype=float)

                estep=EWIN/float(tedist.shape[1])
                tstep=TWIN/float(tedist.shape[0])
        
                clow=0
                chigh=20
                if plotting:
                    fig,axs = plt.subplots(3,4)
    
                    #axs[0][0].set_title('st%.1f, wd%.1f, v%.1f'%(stref,wdref,vref))
                    axs[0][0].pcolor(x)#,vmin=clow,vmax=chigh)
                    #axs[0][0].imshow(x,origin='lower',vmin=clow,vmax=chigh)
                    axs[0][0].set_title('Ximg')

                cmax = 1.0
                cthis = 1.0
                ewidthmean = 1.0
                twidthmean = 1.0
                for i in range(5):
                    indref,rowref,stref,ewidth,vref = scanKernel(wlist,slist,x)
                    if i==0:
                        cmax = vref

                    kern = fillKernel(width=ewidth,strength=stref,kern=kern)
                    twidth=float(TEPROD)/ewidth

                    ewidthmean *= float(i)
                    ewidthmean += ewidth
                    ewidthmean /= float(i+1)
    
                    proj = np.roll(np.roll(kern,-indref,axis=0),rowref,axis=1)
                    #proj = np.roll(np.roll(kern,-indref,axis=0),rowref,axis=1)
                    coeff = np.inner(x.flatten(),proj.flatten())
                    coefflist += [coeff]

                    x -= (coeff*proj).astype(int)
                    proj_display += coeff*proj

                    if plotting:
                        #axs[(i+1)//4][(i+1)%4].imshow(x,vmin=clow,vmax=chigh,origin='lower')
                        axs[(i+1)//4][(i+1)%4].pcolor(x)#,vmin=clow,vmax=chigh)
                        axs[(i+1)//4][(i+1)%4].set_title('rm_%i'%i)
                    

                    temat[(indref + (temat.shape[0]>>1))%temat.shape[0],
                            (rowref + (temat.shape[1]>>1))%temat.shape[1]] += coeff
                    tedist[10:-10,10:-10] = temat
                    blurwidth = int(ewidthmean*2)
                    blurwidth += (blurwidth+1)%2
                    tedist = cv2.GaussianBlur(tedist,(blurwidth,3),0)#,ewidthmean,twidthmean)
                    #tedist = cv2.GaussianBlur(tedist,(13,3),0)#,ewidthmean,twidthmean)
                    tepeaks = detect_peaks(tedist)
                    nsase['pred'] = np.sum(tepeaks)

    
    
                    if plotting:
                        print('ewidthmean = %.2f'%ewidthmean)
                        axs[-1][-4].pcolor(proj_display)
                        axs[-1][-4].set_title('sum projections')
                        #axs[-1][-3].imshow(tedist,origin='lower')
                        #axs[-1][-3].pcolor(tedist)
                        axs[-1][-3].pcolor(tepeaks)
                        axs[-1][-3].set_title('tepeaks, n = %i'%(nsase['pred']))
                        #axs[-1][-2].imshow(temat,origin='lower')
                        axs[-1][-2].pcolor(temat)
                        #axs[-1][-2].imshow(np.roll(np.roll(temat,temat.shape[0]//2,axis=0),temat.shape[1]//2,axis=1),origin='lower')
                        axs[-1][-2].set_title('temat')
                        axs[-1][-1].pcolor(y)
                        #axs[-1][-1].imshow(y,origin='lower')
                        axs[-1][-1].set_title('Ypdf')


                if plotting:
                    plt.show()

                i = min(nsase['true'],o['true'].attrs['hist'].shape[0]-1)
                j = min(nsase['pred'],o['pred'].shape[1]-1)
                o['true'].attrs['hist'][i] += 1
                o['pred'][i,j] += 1
                o['pred'].attrs['hist'][j] += 1
                if i==j:
                    o['pred'].attrs['true'] += 1

                t1=time.time()
                runtimes += [t1-t0]

            o.create_dataset('runtimes',data = np.array(runtimes,dtype=np.float16))

            print('... working coefficient histogram ... ')
            h,b = np.histogram(coefflist,o['coeffhist'].shape[0])
            o['coeffhist'][()] = h
            o['coeffbins'][()] = b

            nbins = 1<<6
            print('... working runtime histogram ... ')
            h,b=np.histogram(runtimes,bins=nbins)
            _=[print('%02i.%i s:'%(int(b[i]),int((b[i]%1)*1e3)) + ' '*v+'+') for i,v in enumerate(h)]
    return

if __name__ == '__main__':
    if len(sys.argv)<2:
        print('give me a file to process')
        tedist = np.zeros((1<<7,1<<7),dtype=float)
        ewidth=2.5
        twidth=float(TEPROD)/ewidth
        estep=EWIN/float(tedist.shape[1])
        tstep=TWIN/float(tedist.shape[0])
        print(tstep,estep,twidth,ewidth,TEPROD/ewidth)
        print(tedist.shape[0]//3,tedist.shape[1]//3)
        addGauss2d(tedist,1,tedist.shape[1]//2,tedist.shape[0]//2,ewidth/estep,twidth/tstep)
        #addGauss2d(tedist,1,tedist.shape[1]//4,tedist.shape[0]//4,10,2)
        print(np.max(tedist))
        plt.imshow(tedist,origin='lower')
        plt.colorbar()
        plt.show()

    else:
        main(sys.argv[1],plotting=False)
