# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:24:23 2021

@author: Pablo
"""


import numpy as np
from time import sleep
from threading import Thread
from datetime import datetime

from pytracker.myArduino import myArduino
from pytracker.myArduino import find_arduinos

import serial.tools.list_ports

class myArduTracker(myArduino):
    def __init__(self, device, baudrate=28800):
        super().__init__(device, baudrate) 

        
        # Initialization
        self.define_pinout()
        self.set_pin_modes()
        self.define_ntc_props()
        
        # ... threads
        self.__setup_pot_link()
        self.__setup_temperature_control()

        
        # Report led status on,
        # ... we recommend to keep the status led blinking even during
        # ... the experiments, to check the PC-Arduino communication.
        # ... If the led stops blinking, communication has been interrupted.
        self.STATUS_LED_PIN = self.pinout['stat1'] 
        self.start_blink_status()
        
        
        
          
    #===============================#
    #     ARDUTRACKER FUNCTIONS     #
    #===============================#  
    def define_ntc_props(self, props=None):
        if props is None:
            self.ntc_props={}
            self.ntc_props['rvd']  = 10_000 # Resistance of the voltage divider in ohm
            self.ntc_props['vd']   =     -1 # Voltage divider configuration, +/-1
            self.ntc_props['rref'] = 10_000 # Resistance of the NTC at the reference temperature in ohm
            self.ntc_props['tref'] =     25 # Reference temperature in ºC
            self.ntc_props['beta'] =   4050 # Beta parameter of the NTC
        else:
            self.ntc_props = props
        
        
        
        
    def define_pinout(self, pinout = None):
        if pinout is None:
            # Create empty pinout dict
            self.pinout ={}
            
            # Digitals
            self.pinout['fan']   = 2
            self.pinout['tec']   = 6     
            self.pinout['stat2'] = 7
            self.pinout['stat1'] = 8
            self.pinout['led1']  = 9
            self.pinout['led2']  = 10
            self.pinout['led3']  = 11
            self.pinout['led13'] = 13    #      step pin of focus stepper motor
            #self.pinout['dir']   = 12   # direction pin of focus stepper motor
            self.pinout['tec_mode']  = 5 # if using DRV8833, low/high turns on cooling/heating modes 
            
            # Analogs
            self.pinout['ntc']   = 0
            self.pinout['pot']   = 1
            
        else:
            self.pinout = pinout
        

            
    def set_pin_modes(self):
        #... digital outputs, BINARY
        self.set_mode( self.pinout['fan'],   'digital_output')
        self.set_mode( self.pinout['stat1'], 'digital_output')
        self.set_mode( self.pinout['stat2'], 'digital_output')
        self.set_mode( self.pinout['tec_mode'], 'digital_output')
        #... PWM
        self.set_mode( self.pinout['tec'],   'pwm')
        self.set_mode( self.pinout['led1'],  'pwm')
        self.set_mode( self.pinout['led2'],  'pwm')
        self.set_mode( self.pinout['led3'],  'pwm')
        #self.set_mode( self.pinout['dir'],   'digital_output')
        self.set_mode( self.pinout['led13'], 'digital_output')
        
        #... analog inputs
        self.set_mode( self.pinout['ntc'],   'analog_input')
        self.set_mode( self.pinout['pot'],   'analog_input')



    def led1(self, value):
        return self.write( self.pinout['led1'], value, mode='pwm')
    
    def led2(self, value):
        return self.write( self.pinout['led2'], value, mode='pwm')

    def led3(self, value):
        return self.write( self.pinout['led3'], value, mode='pwm')
    
    
    def fan(self, value):
        return self.write( self.pinout['fan'], value, mode='digital')
        
    
    def tec(self, value, mode='c'):
        ''' 
            Tec temperatures are clipped.
            - to 150 when cooling, reaching a maximum Troom -  5ºC
            - to  25 when heating, reaching a maximum Troom + 28ºC
        '''
        # If value is null, turn everything off
        if value==0:
            self.write( self.pinout['tec']     , 0, mode='pwm')            
            self.write( self.pinout['tec_mode'], 0, mode='digital')
            return value
        
        # Set working mode heating/cooling
        if mode=='c' or mode=='cool':
            value = np.clip(value, 0, 150)
            self.fan(1)
            self.write( self.pinout['tec']     , value, mode='pwm')
            self.write( self.pinout['tec_mode'],     0, mode='digital')
            return value
        
        elif mode=='h' or mode=='heat':
            value = np.clip(value, 0, 25)
            self.fan(1)
            self.write( self.pinout['tec']     , value, mode='pwm')
            self.write( self.pinout['tec_mode'],     1, mode='digital')
            return value
        
        else:
            print('<E> Incorrect tec mode. It has to be either c(ooling) or h(heating).')
            return
        
    
    
    def pot(self):
        return self.read( self.pinout['pot'] )
            
    
    def temperature(self):
        return self.read( self.pinout['ntc'], **self.ntc_props, mode='temperature' )

        
    #===============================#
    # TEMPERATURE CONTROL FUNCTIONS #
    #===============================#
    def __setup_temperature_control(self):
        self.thread_temperature = Thread()
        self.thread_temperature_running = False
        self.temperature_objective = 20
        self.temperature_maxdiff   = 1
    
    
    def start_temperature_control(self):
        if self.thread_temperature_running:
            print('<W> A previous temperature control thread is running.')
            print('   Please stop it before starting another.')
            return
        else:
            self.thread_temperature = Thread(target= self.__threadfun_temperature_control , daemon = True)
            self.thread_temperature.start()
    
    
    def __threadfun_temperature_control(self):
        self.thread_temperature_running = True
                    
        DTIME          = 0.2        
        tLast          = datetime.now()
        
        while self.thread_temperature_running:    
            if (datetime.now()-tLast).total_seconds() > DTIME:
                
                # Update current temperature
                tnow = self.temperature()
                
                # Check if temperature is within range
                if np.abs(tnow-self.temperature_objective)> self.temperature_maxdiff:
                    self.write( self.pinout['stat2'], 1, mode='digital')
                else:
                    self.write( self.pinout['stat2'], 0, mode='digital')
                
                # Update tLast and sleep
                tLast = datetime.now()
                sleep(0.02)
        
                
    #===============================#
    #       POTLINK FUNCTIONS       #
    #===============================#
    def __setup_pot_link(self):
        self.thread_pot_link = Thread()
        self.thread_pot_link_running = False
    
    def start_pot_link(self, output='led1'):
        if self.thread_pot_link_running:
            print('<W> A previous pot_link thread is running.')
            print('   Please stop it before starting another.')
            return
        else:
            self.thread_pot_link = Thread(target= self.__threadfun_potlink , args=(output,) , daemon = True)
            self.thread_pot_link.start()
    
    
    def __threadfun_potlink(self, output):
        self.thread_pot_link_running = True
            
        DTIME          = 0.1        
        NVALS          = 50
        tLast          = datetime.now()
        print('Correctly connected pot to %s' % output)
        while self.thread_pot_link_running:    
            if (datetime.now()-tLast).total_seconds() > DTIME:
                value = np.round( self.pot()*NVALS )/NVALS*255
                self.write( self.pinout[output], int(value), mode='pwm')
                
                tLast = datetime.now()
                sleep(0.02)
    

    




