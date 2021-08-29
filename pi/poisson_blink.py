# poisson blink
# Blink LED on pin 8 at random intervals
# JMA 8 Aug 2021

import RPi.GPIO as gpio
from time import sleep
import sys
from math import *
from random import *

pin =23 # VIA pin 24, Rpi pin 28 ?? 

gpio.setwarnings(True)
gpio.setmode(gpio.BOARD)
# polarity LOW and HIGH are reversed
gpio.setup(pin, gpio.OUT, initial=gpio.LOW)

while True:
    rexp =  min(100, expovariate(0.4))
    print('    ', rexp,  file=sys.stderr)
    gpio.output(pin, gpio.HIGH)
    sleep(rexp)
    hexp = min(100, expovariate(1.1))
    print(hexp, file=sys.stderr)
    gpio.output(pin, gpio.LOW)
    sleep(hexp)

    
