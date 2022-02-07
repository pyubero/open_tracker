# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:24:23 2021

@author: Pablo
"""

from pymata4 import pymata4
import serial.tools.list_ports
import numpy as np
from time import sleep
import threading
from threading import Thread
from datetime import datetime


class myArduino:
    def __init__(self, device):
        print('Initializing arduino...')
        self.board = pymata4.Pymata4(com_port = device, baud_rate = 57600)
        
        self.total_analog_pins = len(self.board.analog_pins)
        self.total_digital_pins = len(self.board.digital_pins) - self.total_analog_pins
        
        # As with pymata4 the state of output pins cannot be queried, we store 
        # ... their values as they change.
        self.digitalPins = np.zeros( ( self.total_digital_pins,) )
        self._blinking_running = [False    for _ in range(self.total_digital_pins) ]
        self.blink_threads     = [Thread() for _ in range(self.total_digital_pins) ]
        
    def __del__(self):
        self.close()
         
     
        
    def set_mode(self, pin_number, mode, callback = None):
        if mode=='analog_input':
            self.board.set_pin_mode_analog_input(pin_number, callback=callback)
            
        elif mode=='digital_input':
            self.board.set_pin_mode_digital_input(pin_number, callback=callback)
            
        elif mode=='digital_input_pullup':
            self.board.set_pin_mode_digital_input_pullup(pin_number, callback=callback)
        
        elif mode=='digital_output':
            self.board.set_pin_mode_digital_output(pin_number)    

        elif mode=='pwm':
            self.board.set_pin_mode_pwm_output(pin_number)    
        
        elif mode=='servo':
            self.board.set_pin_mode_servo(pin_number)


    def read(self, pin_number, mode='analog',**kwargs):
        '''Here kwargs can be Rvd, Vd, Rref, Tref, Iref'''
        
        if mode =='analog':
            return self.board.analog_read(pin_number)[0]/1023
        
        elif mode =='digital':
            return self.board.digital_read(pin_number)[0]
        
        elif mode =='voltage_divider':
            return a2r( self.read(pin_number, mode='analog'), **kwargs)
        
        elif mode =='temperature':
            return r2t( self.read(pin_number, mode='voltage_divider', **kwargs), **kwargs)
            
        elif mode =='illuminance':
            return r2i( self.read(pin_number, mode='voltage_divider', **kwargs), **kwargs)
        
        
        
    def write(self, pin_number, value, mode='digital'):
        # Store current pin state in our own array
        self.digitalPins[pin_number] = value
        
        # ... and then write value
        if mode =='digital':         
            return self.board.digital_write(pin_number, value)
        
        elif mode=='pwm':
            return self.board.pwm_write( pin_number, value)
        
        elif mode=='servo':
            return self.board.servo_write(pin_number, value)
    
    
    def close(self):
        #Disable all analog and digital reportings
        for pin in range( self.total_analog_pins):
            self.board.disable_analog_reporting(pin)
            sleep(0.1)
        
        for pin in range( self.total_digital_pins):
            self.board.disable_digital_reporting(pin)
            sleep(0.1)
        
        
        # Turn off all digital pins
        for pin in range(self.total_digital_pins):
            self.write(pin, 0)
            sleep(0.1)
        
        
        self.board.shutdown()
        print('Arduino communication closed.')    

        
    def write_pwm(self, pin_number, value):
        return self.board.pwm_write( pin_number, value)


    def write_servo(self, pin_number, angle=0):
        return self.board.servo_write( pin_number, angle)


    
    def toggle(self, pin_number):
        pin_mode = self.board.get_pin_state(pin_number)[1]
        
        if pin_mode == 1: #DIGITAL OUTPUT
            new_value = 1.0 - self.digitalPins[pin_number]
            self.write( pin_number, new_value, mode='digital')
            
        elif pin_mode == 3: #DIGITAL PWM
            new_value = 255 - self.digitalPins[pin_number]
            self.write( pin_number, new_value, mode='pwm')
            
        elif pin_mode == 4: #DIGITAL SERVO
            new_value = 180 - self.digitalPins[pin_number]
            self.write( pin_number, new_value, mode='servo')
        
        else:
            print('<< E >> myArduino.toggle(pin_number) needs a valid digital output pin number')
            return
        
        
    # def blink_start(self, pin_number, frequency):
    #     if self._blinking_running[pin_number]:
    #         self._blink_stop(pin_number)
        
    #     self.blink_threads[pin_number] = Thread( target= lambda: self._blink_threadfun(pin_number, frequency), daemon = True)
    #     self.blink_threads[pin_number].start()
        
        
        
    # def _blink_stop(self, pin_number):
    #     if self._blinking_running[pin_number]:
    #         self._blinking_running[pin_number] = False
    #         self.blink_threads[pin_number].join()
    #         self.write(pin_number, 0)

        
    # def _blink_threadfun(self, pin_number , frequency = 1.0 ):         
    #     self._blinking_running[pin_number] = True
        
    #     dtime = 1.0/frequency 
    #     print('Blinking thread started.')
        
    #     tLast = datetime.now()
    #     while self._blinking_running[pin_number]:    
            
    #         if (datetime.now()-tLast).total_seconds() > dtime:
    #             self.toggle(pin_number)
    #             tLast = datetime.now()
    #             sleep(0.01)

        
    #     print('Blinking thread stopped.')






##########################################################
################# SOME UTILITY FUNCTIONS #################
def find_devices():
    return [comport.device for comport in serial.tools.list_ports.comports()]
     

def a2r( A , rvd = 10_000, vd=+1, **kwargs):
    '''This function returns the resistance measured at a voltage divider in units of Rvd.
    The voltage divider configuration is selected with Vd=+1 or Vd=-1'''
    rvd = float(rvd)
    vd  = float(vd)
    if A==0:
        return 1e-6
    else:
        return rvd*( 1/A -1 )**vd


def r2t( R , rref=10_000, tref=25, beta=4050, **kwargs):
    '''This functions converts the resistance of an NTC into a temperature (ºC)'''
    rref = float(rref)
    tref = float(tref)
    beta = float(beta)
    return (1/(tref+273) + np.log(R/rref)/beta)**-1 -273

    
def r2i( R, rref=10_000, iref= 1000, gamma = 0.65, **kwargs):
    '''This functions converts the resistance of an LDR into luminance (lux)'''
    rref = float(rref)
    iref = float(iref)
    gamma= float(gamma)
    
    return iref*(R/rref)**(-gamma)
    

# def program_function(delay, function ):
#     start_time = threading.Timer( interval=delay, function=function )

    
    
    

    




