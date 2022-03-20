#!/bin/bash
while true
do
    /usr/bin/python3  poll_trace.py | tee -a poll_trace.txt
    # /usr/local/bin/python3.10  poll_trace.py | tee -a poll_trace.txt
    sleep 120
done

    
