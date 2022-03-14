# poll trace py
# periodically run traceroute and ping to monitor internet service
# JMA 6 Aug 2021

import pickle, os, os.path, sys, subprocess
import pprint, re, string, time
import subprocess
import datetime

PING_TARGET = '8.8.8.8'
TRACEROUTE_TARGET = 'be-232-rar01.santaclara.ca.sfba.comcast.net'

dbg      = False


########################################################################
def iterate_monitors(target):
    err = subprocess.Popen(['/bin/ping', '-c1', '-D', target],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    proc_out, proc_err = err.communicate()
    proc_out = proc_out.decode('utf-8').splitlines()
    proc_err = proc_err.decode('utf-8').splitlines()
    if dbg:
        print('out: ',proc_out)
        print('err: ',proc_err)
    return [proc_out, proc_err]

##############################################################
def iterate_traceroute(target):
    err = subprocess.Popen(['/usr/sbin/traceroute', target],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    proc_out, proc_err = err.communicate()
    proc_out = proc_out.decode('utf-8').splitlines()
    proc_err = proc_err.decode('utf-8').splitlines()
    if dbg: 
        print('out: ',proc_out)
        print('err: ',proc_err)
    return [proc_out, proc_err]

    
def run_monitors(ptarget=PING_TARGET, ttarget=TRACEROUTE_TARGET ):
    millisec_responses = {}
    ping_start = time.time()
    m = iterate_monitors(ptarget)
    ping_end =  time.time()
    ping_runtime = 1000 * (ping_end - ping_start)
    tr = iterate_traceroute(ttarget)
    trace_runtime = 1000 * (time.time() - ping_end)
    ping_summary = m[0][-1]
    trace_summary = tr[0][-1]
    ping_duration = re.split(r'/',ping_summary)
    trace_duration = re.split(r'\s+ms', trace_summary)
    if len(ping_duration) > 1:
        millisec_responses = dict(ping_runtime = f'{ping_runtime:.3f}',
                                  ping_duration = ping_duration[-2],
                                  trace_runtime = f'{trace_runtime:.3f}',
                                  trace_duration = trace_duration[-2].strip())
    millisec_responses['time_stamp'] = re.sub(' ', '_', datetime.datetime.now().isoformat())
    return millisec_responses

if __name__ == '__main__':

    print(run_monitors())

    
