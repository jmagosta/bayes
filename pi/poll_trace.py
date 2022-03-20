# poll trace py
# periodically run traceroute and ping to monitor internet service
# JMA 6 Aug 2021

import pickle, os, os.path, sys, subprocess
import pprint, re, string, time
from pathlib import Path
import subprocess
import datetime
from blinker import Blink

PING_TARGET = '8.8.8.8'
TRACEROUTE_TARGET = 'be-232-rar01.santaclara.ca.sfba.comcast.net'
LOG_FILE_PATH = '/media/pi/Elements/netlog'
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

##############################################################
def is_intermittent(responses):
    'Did the test indicate bad network performance'
    bad_pr = float(responses['ping_runtime']) > 50
    bad_pd = float(responses['ping_duration']) > 30
    bad_tr = float(responses['trace_runtime']) > 500
    bad_td = float(responses['trace_duration']) > 0
    return bad_pr or bad_pd or bad_tr or bad_td

##############################################################
def run_monitors(ptarget=PING_TARGET, ttarget=TRACEROUTE_TARGET ):
    millisec_responses = {}
    trace_summary = ''
    ping_start = time.time()
    m = iterate_monitors(ptarget)
    ping_end =  time.time()
    ping_runtime = 1000 * (ping_end - ping_start)
    tr = iterate_traceroute(ttarget)
    trace_runtime = 1000 * (time.time() - ping_end)
    b= Blink()
    bad_network = False 
    # print('m', m)
    if len(m[0]) > 0 and len(tr[0]) > 0:
        ping_summary = m[0][-1]
        ping_duration = re.split(r'/',ping_summary)
        trace_summary = tr[0][-1]
        trace_duration = re.split(r'\s+ms', trace_summary)
        millisec_responses = dict(ping_runtime = '{ping_runtime:.3f}'.format(ping_runtime=ping_runtime),
                                  ping_duration = ping_duration[-2] if len(ping_duration) >1 else 'NaN',
                                  trace_runtime = '{trace_runtime:.3f}'.format(trace_runtime=trace_runtime),
                                  trace_duration = trace_duration[-2].strip())
        bad_network = is_intermittent(millisec_responses)
    else:
        millisec_responses['msg'] = m[1] if len(m) >0 else 'ping failed'
        bad_network = True
    millisec_responses['time_stamp'] = re.sub(' ', '_', datetime.datetime.now().isoformat())
    if bad_network:
        b.on()
    else:
        b.off()
        # b.cleanup()
    return millisec_responses

if __name__ == '__main__':

    if dbg:
        print(run_monitors())
    else:
        with open(LOG_FILE_PATH+'/poll_trace', 'a') as log_fd:
            print(run_monitors(), file= log_fd)
            
