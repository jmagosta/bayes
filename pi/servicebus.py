# code from https://azure.microsoft.com/en-us/documentation/articles/service-bus-python-how-to-use-queues/
#!/usr/bin/python
# 14 Dec 2015  JMA
# servicebus.py
# Post live data from the TED5000 to Azure service bus
"""
Collect voltage and wattage data from the TED api. 
And send each reading when read to Azure.


"""
import pprint, os, sys
import datetime
from time import sleep, time, strptime
import argparse
from json import dumps, loads
from azure.servicebus import ServiceBusService, Message, Queue
import poll_ted as pt

dbg = False

#####################################################################
CONFIG = loads(open('config.json').read())

def bus_setup():
    # Create a ServiceBusService object.
    # Replace mynamespace, sharedaccesskeyname, and sharedaccesskey 
    # with your namespace, shared access signature (SAS) key name, and value.
    bus_service = ServiceBusService('TEDnamespace',
                                    shared_access_key_name=CONFIG['POLICY_NAME'],
                                    shared_access_key_value=CONFIG['PRIMARY_KEY'])
    return bus_service


########################################################################
def convert_to_dict(reading):
        # Express the readings as a labeled dictionary.  In the future, do this instead
        # in poll_ted.parse_ted_history_xml
        return {"sensed":reading[0], "watts":reading[1], "decivolts":reading[2]}


########################################################################
# Write the output to a tab-delimited file, 
# And return the output as a list of lists
def read_ted(interval, starttime, duration, out_file=pt.out_filename('SB')):
    then = starttime + duration
    duration = duration.total_seconds()
    uri = pt.request_uri(CONFIG['PERIOD'])
    bus = bus_setup()
    extract_seq = [None]
    with open(out_file, 'wb') as out_fd:
        print('Begun polling.', out_file)
        for reading, extract_seq in pt.iterate_readings(interval, CONFIG['PERIOD'], uri, extract_seq):
            rec = dumps(convert_to_dict(reading))
            extract_seq.append(rec)
            pt.wr_list(out_fd, reading)
            try:
                bus.send_event(CONFIG['EVENTHUB_NAME'], rec)
            except:
                print('send_event failed for ', rec)
            if datetime.datetime.today() >= then and duration != 0:
                break
            
    return extract_seq[1:]

#####################################################################
def options():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('interval', nargs='?', default=1,
                            help='Number of seconds between checks for readings. An integer > 0.')
        parser.add_argument('duration', nargs='?', default=CONFIG['DURATION'],
                            help='Run for this length of time as hr:min:sec., eg 00:00:60 for 60 seconds.')

        parser.add_argument('-v', action='store_true',
                            help='Print verbose output of intermediate results.')
        return parser.parse_args()

        
#####################################################################
def main(intv, now, dur):
    return read_ted(intv, now, dur)


########################################################################
if __name__ == '__main__':
    ## If invoked with no args, just print the usage string
    args = options()
    dbg = args.v
    pt.dbg = dbg

    now = datetime.datetime.today()
    duration= strptime(args.duration, '%H:%M:%S')
    duration = datetime.timedelta(hours=duration.tm_hour, minutes=duration.tm_min, seconds = duration.tm_sec)
    pprint.pprint(main(int(args.interval), now, duration))
    print(dbg, sys.argv, "Done in ", '%s' % (datetime.datetime.today() - now), " hr:min:sec.",  file=sys.stderr)


### (c) 2015 John M Agosta


