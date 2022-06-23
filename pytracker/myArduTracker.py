# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:24:23 2021

@author: Pablo
"""

from pymata4 import pymata4
import serial.tools.list_ports
import numpy as np
from time import sleep
from threading import Thread
from datetime import datetime


class myArduTracker:
    def __init__(self, device):
        print('Initializing arduino...')
        self.board = pymata4.Pymata4(com_port = device, baud_rate = 28800)
        
        self.total_analog_pins = len(self.board.analog_pins)
        self.total_digital_pins = len(self.board.digital_pins) - self.total_analog_pins
        
        # As with pymata4 the state of output pins cannot be queried, we store 
        # ... their values as they change.
        self.digitalPins = np.zeros( ( self.total_digital_pins,) )
        self._blinking_running = [False    for _ in range(self.total_digital_pins) ]
        self.blink_threads     = [Thread() for _ in range(self.total_digital_pins) ]
        
        
        # Initialization
        self.define_pinout()
        self.set_pin_modes()
        self.define_ntc_props()
        
        #... threads
        self.__setup_pot_link()
        self.__setup_temperature_control()
        self.__setup_blinking_leds()

        
        # Report led status on,
        # ... we recommend to keep the status led blinking even during
        # ... the experiments, to check the PC-Arduino communication.
        # ... If the led stops blinking, communication has been interrupted.
        # self.start_blink_status()
        
        
        
        
    def __del__(self):
        self.close()
         
     
    #===============================#
    #     BASIC PYMATA FUNCTIONS    #
    #===============================#
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
    
    
    def close(self, analog=None , digital=None ):
        # Stop all threads
        if self.thread_led_status_running:
            self.thread_led_status_running = False
            self.thread_led_status.join() 
            
        if analog is None:
            analog = range( self.total_analog_pins)
            
        if digital is None:
            digital = range( self.total_digital_pins)
            
            
        # Disable all analog and digital reportings
        for pin in analog:
            self.board.disable_analog_reporting(pin)
            sleep(0.1)
        
        for pin in digital:
            self.board.disable_digital_reporting(pin)
            sleep(0.1)
        
        
        # Turn off all digital pins
        for pin in digital:
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
    
    
    
    #===============================#
    #    BLINKING LEDS FUNCTIONS    #
    #===============================#
    def __setup_blinking_leds(self):
        self.thread_led_status = Thread()
        self.thread_led_status_running = False
        
        
    def __blink_threadfun_status(self):
        self.thread_led_status_running = True
            
        DTIME          = 0.3        
        STATUS_LED_PIN = self.pinout['stat1']
        tLast          = datetime.now()
        
        while self.thread_led_status_running:    
            if (datetime.now()-tLast).total_seconds() > DTIME:
                self.toggle(STATUS_LED_PIN)
                tLast = datetime.now()
                sleep(0.02)
                
    def start_blink_status(self):
        if self.thread_led_status_running:
            return
        else:
            self.thread_led_status = Thread(target= self.__blink_threadfun_status , daemon = True)
            self.thread_led_status.start()
                
            
        
        
        
        
 





##########################################################
################# SOME UTILITY FUNCTIONS #################
def find_devices(filter_name =None , filter_value =None):
    comports = serial.tools.list_ports.comports()
    
    if filter_value is not None:
        if filter_name == 'pid':
            return [ port.name for port in comports if port.pid == filter_value ]
        if filter_name == 'vid':
            return [ port.name for port in comports if port.pid == filter_value ]
        else:
            print('<W> Please introduce a valid filter_name either "vid" or "pid".')
            return 
    else:
        return [comport.device for comport in serial.tools.list_ports.comports()]
      
def find_arduinos( filter_name = 'pid', filter_value = 29987):
    return find_devices( filter_name=filter_name, filter_value=filter_value)


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
    

    
    

    




