#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py


def main(fnames):
    with h5py.File(fnames[0],'r') as f:
        shape = f['pred'].shape
    confusion = np.zeros(shape,dtype=np.uint16)
    for fname in fnames:
        with h5py.File(fname,'r') as f:
            confusion += f['pred'][()]

    truehist = np.sum(confusion,axis=1)
    normmat = np.tile(truehist,(len(truehist),1)).astype(np.float16).T
    _=[print(' '*int(v) + '+') for v in truehist]
    inds = np.where(normmat>0)
    print(np.sum(confusion))
    normconfusion = np.copy(confusion).astype(np.float16)
    normconfusion[inds] /= normmat[inds]
    normconfusion[0,0] = 1

    x = [i for i in range(5)]
    y = [i for i in range(5)]
    x += [i for i in range(4)]
    y += [i+1 for i in range(4)]
    x += [i+1 for i in range(4)]
    y += [i for i in range(4)]
    x += [i for i in range(3)]
    y += [i+2 for i in range(3)]
    x += [i+2 for i in range(3)]
    y += [i for i in range(3)]
    x += [i for i in range(2)]
    y += [i+3 for i in range(2)]
    x += [i+3 for i in range(2)]
    y += [i for i in range(2)]
    x += [0]
    y += [4]
    x += [4]
    y += [0]
    #x += [i for i in range(5)]
    #y += [i+5 for i in range(5)]
    #x += [i+5 for i in range(5)]
    #y += [i for i in range(5)]
    '''
    x = [i+1 for i in range(5)]
    y = [i+1 for i in range(5)]
    x += [i+1 for i in range(5)]
    y += [i+2 for i in range(5)]
    x += [i+2 for i in range(5)]
    y += [i+1 for i in range(5)]
    x += [i+1 for i in range(5)]
    y += [i+3 for i in range(5)]
    x += [i+3 for i in range(5)]
    y += [i+1 for i in range(5)]
    x += [i+1 for i in range(5)]
    y += [i+4 for i in range(5)]
    x += [i+4 for i in range(5)]
    y += [i+1 for i in range(5)]
    x += [i+1 for i in range(5)]
    y += [i+5 for i in range(5)]
    x += [i+5 for i in range(5)]
    y += [i+1 for i in range(5)]
    '''

    annotations = []
    for i in range(len(x)):
        annotations += ["%0.1f %%"%(float(normconfusion[y[i],x[i]]*100))]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    
    ax.imshow(normconfusion[:5,:5])
    #ax.imshow(normmat[:6,:6])
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for xi, yi, txt in zip(x, y, annotations):

        if normconfusion[yi,xi]>.4:
            plt.text(xi, yi, txt, fontdict=None, color='xkcd:dark blue', fontsize = 'large', fontweight = 'heavy', ha = 'center', va = 'center')
        else:
            plt.text(xi, yi, txt, fontdict=None, color='xkcd:eggshell', fontsize = 'large', fontweight = 'heavy', ha = 'center', va = 'center')
        '''
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(-1.5, -1.5), textcoords='offset points')
                '''
    plt.savefig('./figs/plotConfusions_redeux.png')
    plt.show()
    with h5py.File('./figs/plotConfusions_redeux.h5','w') as ofile:
        ofile.create_dataset('confusions',data=normconfusion,dtype=float)

    '''
    plt.imshow(normconfusion,origin='lower')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.colorbar()
    plt.show()
    '''
    return

if __name__ == '__main__':
    if len(sys.argv)<2:
        print('./src/plotConfusion.py <list of confusion.h5 files>')
    else:
        filenames = [str(name) for name in sys.argv[1:]]
        main(filenames)
