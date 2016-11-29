#!/usr/bin/env python

from os import listdir # directory list module
import os,sys,argparse # OS operations



#############################
##### MAIN BODY OF CODE #####
#############################

def main(flocation):

    flist=[f for f in listdir(flocation) if f[-5:]=='.fits']

    for f in flist:
        fname=f[:-5]
        ftorque=open('torque/'+fname+'.torque','w')
        ftorque.write('#!/bin/bash \n')
        ftorque.write('\n')
        ftorque.write('#PBS -o /data3a/work/jspeagle/frankenz/torque/outputs/${PBS_JOBNAME}.out \n')
        ftorque.write('#PBS -e /data3a/work/jspeagle/frankenz/torque/outputs/${PBS_JOBNAME}.err \n')
        ftorque.write('#PBS -l nodes=1 \n')
        ftorque.write('#PBS -q tiny \n')
        ftorque.write('#PBS -l walltime=200:00:00 \n')
        ftorque.write('#PBS -N frankenz_'+fname+' \n')
        ftorque.write('#PBS -m bea \n')
        ftorque.write('#PBS -M jspeagle@cfa.harvard.edu \n')
        ftorque.write('#PBS -V \n')
        ftorque.write('\n')
        ftorque.write('export PATH="/data1a/ana/anaconda/bin:$PATH" \n')
        ftorque.write('cd /data3a/work/jspeagle/frankenz/ \n')
        ftorque.write('python frankenz_script_s16a.py config/frankenz.config '+flocation+f+' '+fname+' ../s16a/')
        ftorque.close()
        os.system('qsub '+'torque/'+fname+'.torque')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('location',help="S16A target catalog location")
    args = parser.parse_args()

    main(args.location)
