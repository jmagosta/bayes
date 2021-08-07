# poll trace py
# periodically run traceroute and ping to monitor internet service
# JMA 6 Aug 2021

import pickle, os, os.path, sys, subprocess
import pprint, re, string, time
import subprocess

POLLING_INTERVAL = 60 * 5 # seconds
PING_TARGET = '8.8.8.8'
TRACEROUTE_TARGET = 'be-232-rar01.santaclara.ca.sfba.comcast.net'

dbg      = False
poll     = True


########################################################################
def iterate_monitors(interval=POLLING_INTERVAL):
    err = subprocess.Popen(['/bin/ping', '-c1', '-D', PING_TARGET],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    proc_out, proc_err = err.communicate()
    proc_out = proc_out.decode('utf-8').splitlines()
    print('out: ',proc_out)
    print('err: ',proc_err)

##############################################################
def iterate_traceroute(interval=POLLING_INTERVAL):
    err = subprocess.Popen(['/usr/sbin/traceroute', TRACEROUTE_TARGET],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    proc_out, proc_err = err.communicate()
    proc_out = proc_out.decode('utf-8').splitlines()
    print('out: ',proc_out)
    print('err: ',proc_err)
    

iterate_monitors()
iterate_traceroute()

    

