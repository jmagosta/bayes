# run TED measurements every ten seconds for a day
# JMA 21 Dec 2015
# First check the disk space available !
df
# Then run for one day. 
python servicebus.py 10 '23:59:59'

