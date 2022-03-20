# poisson blink
# Blink LED on pin 8 at random intervals
# JMA 8 Aug 2021 / 20 Mar 2022

import RPi.GPIO as gpio
from time import sleep
import sys
from math import *
from random import *

class Blink (object):

    def __init__(self) -> None:
        self.pin =23 # VIA pin 24, Rpi pin 28 ?? 
        gpio.setwarnings(True)
        gpio.setmode(gpio.BOARD)
        # polarity LOW and HIGH are reversed
        gpio.setup(self.pin, gpio.OUT, initial=gpio.LOW)

    def on(self):
        gpio.output(self.pin, gpio.HIGH)

    def off(self):
        gpio.output(self.pin, gpio.LOW)

if __name__ == '__main__':
    b = Blink()
    while True:
        rexp =  min(100, expovariate(0.4))
        print('    ', rexp,  file=sys.stderr)
        gpio.output(b.pin, gpio.HIGH)
        sleep(rexp)
        hexp = min(100, expovariate(1.1))
        print(hexp, file=sys.stderr)
        gpio.output(b.pin, gpio.LOW)
        sleep(hexp)

    
