#!/usr/bin/python3

import h5py
from simulation import Params, runprocess
import sys
import multiprocessing as mp
import os
import re

import argparse

parser = argparse.ArgumentParser(description='CookieBox simulator for Attosecond Angular Streaking')
parser.add_argument('-ofname', type=str,required=True, help='ouput path and base file name')
parser.add_argument('-n_threads',   type=int, default=2, help='Number of Threads')
parser.add_argument('-n_angles',type=int, default=128, help='Number of angles')
parser.add_argument('-n_images', type=int,default=10, help='Number of images per thread')
parser.add_argument('-drawscale', type=int,default=2,required=False, help='Scaling for draws from the distribution, e.g. scale the number of electrons')
parser.add_argument('-testsplit', type=float,default=0.1,required=False, help='test images as percent of total')

def main():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)
    m = re.search('(^.*)\/(\w+)\.h5',args.ofname)
    if not m:
        print('failed filename match for ofname = %s'%args.ofname)
        return
    print('%s\t%s'%(m.group(1),m.group(2)))

    if not os.path.exists(m.group(1)):
        os.makedirs(m.group(1))
    paramslist = [Params(m.group(1),m.group(2),args.n_images) for i in range(args.n_threads)]
    for p in paramslist:
        p.setnangles(args.n_angles).setdrawscale(args.drawscale).settestsplit(args.testsplit)

    with mp.Pool(processes=len(paramslist)) as pool:
        pool.map(runprocess,paramslist)

    return

if __name__ == '__main__':
    main()
