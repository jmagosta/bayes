#!/usr/bin/python
# 14 Oct 2013  JMA
# poll_ted.py
# Collect live data from the TED5000
"""
Package to collect voltage and wattage data from the TED api. 
As an application, it calls the TED5000 repeatedly and prints out
lists as 
[datetime, voltage * 10, wattage ]

Usage:
% python poll_ted.py [-v] interval [repetitions]

"""
from __future__ import division
import pickle, os, os.path, sys, subprocess
import pprint, re, string, time
import urllib3


dbg      = False
poll     = True

uri_ip = '192.168.4.24'
uri_path = '/history/'
uri_suffix = 'history.xml'
uri_args = '?MTU=0&COUNT=2&INDEX=10'
repetitions = 3600
interval = 1
PERIOD = 'second'

########################################################################
def tags_on_lines(xml_str):
    ## Split xml so that each tag occupies a separate line
    return re.sub(r'</(\w+)><(\w+)>', r'</\1>\r\n<\2>', xml_str)


########################################################################
def extraction(ex, cnt, the_date, the_power, the_voltage):
    if the_date:
        if dbg: print(cnt,'\t', the_date,'\t', int(the_power)/1000.0,'\t', int(the_voltage)/10.0)
        ex.append( [the_date, int(the_power), int(the_voltage)] )
    return ex
 
    
########################################################################
def parse_ted_history_xml(lines, period):
    ## xml is simple, just iterate thru the page, ignore nesting
    ## but don't assume that the tags are on separate lines. 
    period_pat = re.compile(r'<'+period+'>', re.IGNORECASE)
    meas_pat = re.compile(r'<(date|power|voltage)>([^<]+)', re.IGNORECASE)
    meas_ct = 0
    date, power, voltage = None, None, None
    extracted = []
     
    for a_line in lines:
        ## find the next reading set. 
        m = re.match(period_pat, a_line)
        if m:
            #if dbg: 
            #    print meas_ct, a_line
            extracted = extraction(extracted, meas_ct, date, power, voltage)
            meas_ct +=1

        d = re.match(meas_pat, a_line)
        if d:
            meas_name = d.group(1).upper()
            if dbg: print(meas_ct, meas_name, d.group(2))
            if meas_name == 'DATE':
                date = d.group(2)
            if meas_name == 'POWER':
                power = d.group(2)
            if meas_name == 'VOLTAGE':
                voltage = d.group(2)
    extracted = extraction(extracted, meas_ct, date, power, voltage)
    return extracted


########################################################################
def extract_ted(uri, the_period):
    #if not the_period in period_options:
    #    print 'Error ', the_period, ' must be in ', period_options
    ## Retrieve xml
    try:
        http = urllib3.PoolManager()
        xml_lines = http.request('GET', uri).data.decode('utf-8')
    except Exception as  e:
        print(e, 'url failed: ', uri, file=sys.stderr)
        return None
    ## Convert XML to list of lines.
    xml_lines = tags_on_lines(xml_lines)
    xml_lines = xml_lines.split('\r\n')
    if dbg: 
        print('XML ==========================')
        pprint.pprint(xml_lines)
        print()
    ## extract records from xml
    return parse_ted_history_xml(xml_lines, period=the_period)




########################################################################
def wr_list(fd, the_list):
    fmt_list = [str(x) for x in the_list]
    print(fmt_list, file=sys.stderr)
    record = '\t'.join(fmt_list) + '\n'
    fd.write(record.encode('utf-8'))

    
########################################################################
def file_parse(args):

    xml_lines = []
    try:
        with open(args[0]) as fd:
            for a_line in fd.readlines():
                xml_lines.append(a_line[:-1])
    except:
        e = sys.exc_info()[1]
        print(e, '')
        return None

    parse_ted_history_xml(xml_lines, period='second')


########################################################################
def iterate_readings(interval, period, uri, seq): 
    last_extract = []
    #for z in range(reps):
    cnt = 0
    while True:
        extract = extract_ted(uri, period)
        # Check for overlap in the last two records returned, 
        # to avoid duplicates.
        time.sleep(interval - 0.1)
        if dbg:
            print('slept ', cnt)
            cnt +=1
        if extract and extract != last_extract:
            if extract[1] != seq[-1]:
                extracted = extract[1]
            if extract[0] != seq[-1]:
                extracted = extract[0]
            last_extract = extract
            yield (extracted, seq)


########################################################################
def request_uri(period= 'second'):
    ## request url
    uri = 'http://' + uri_ip + uri_path  + period + uri_suffix + uri_args
    if dbg: print('uri: ' + uri)
    return uri


########################################################################
# Write the output to a tab-delimited file, 
# And return the output as a list of lists
def read_ted(interval, period, repetitions, out_file):
    uri = request_uri(period)
    extract_seq = [None]
    with open(out_file, 'wb') as out_fd:
        print( 'Begun polling.', out_file)
        count = 0
        for reading, extract_seq in iterate_readings(interval, period, uri, extract_seq):
            extract_seq.append(reading)
            wr_list(out_fd, reading)
            if dbg:
                print(count, ':', end= '')
            if count >= repetitions:
                break
            count += 1

    return extract_seq

#######################################################################
def out_filename(s = ''):
    return 'ted'+s+time.strftime('%m-%d_%H-%M-%S')+'.log'

#######################################################################
def main2(interval, out_file = 'ted500.out'):
    return read_ted(interval, PERIOD, repetitions, out_file)

########################################################################
if __name__ == '__main__':
	
    
## If invoked with no args, just print the usage string
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(-1)

    if '-v' in sys.argv:
        del(sys.argv[sys.argv.index('-v')])
        dbg = True

    args = sys.argv[1:]
    if len(args) > 1:
        repetitions = int(args[1]) -1
    elif len(args) > 0:
        interval = int(args[0])
    
    start = time.time()
    outfn = out_filename()
    lofl = main2(interval, outfn)
    pprint.pprint(lofl)
 
    print(dbg, sys.argv, "Done in ", '%5.3f' % (time.time() - start), " secs!", file=sys.stderr)


### (c) 2015 John M Agosta


