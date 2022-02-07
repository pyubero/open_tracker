# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:27:39 2021

@author: Pablo
"""

from pymata4 import pymata4
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import QTimer
import serial.tools.list_ports
import sys
from time import sleep
import numpy as np
import random

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



def find_devices():
    return [comport.device for comport in serial.tools.list_ports.comports()]
     


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=72):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        
        
        
class Ui(QtWidgets.QMainWindow):        
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('easyDuino_gui.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.statusbar.showMessage('Welcome to the easy arduino interface!')
    
        self.board = None
        #self.load_comports()
        self.list_comports.addItem('Please load')
        self.arduino_interface(False)
        
        self.max_historic_data = 120
        self.timer_update_time     = 0.500 #in seconds
        self.historic_data = np.zeros( (8,1) )
        self.new_data_vector= np.zeros( (8,1) )
        self.time_vector    = np.zeros( (1,1) )
        
        #PLOTS
        self.plot_canvas1 = MplCanvas(self,dpi=100)
        self.plot_layout1.addWidget(self.plot_canvas1)
        print( self.plot_canvas1.axes.get_position())
        self.plot_canvas1.axes.set_position( (0.15, 0.25, 0.6, 0.7 ) )
        # self.plot_canvas1.axes.legend( ('A0','A1','A2','A3','A4','A5'), bbox_to_anchor=(0.85, 0.1, 0.5, 1) )
        
        
        
        # timer to update analog readings
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_analog_values)


        #... push buttons
        self.pb_reload_comports.clicked.connect( self.load_comports )
        self.pb_connect.clicked.connect( self.connect )

        
        #... LISTS
        self.analog_checkboxes= [self.chkb_a0, self.chkb_a1, self.chkb_a2, self.chkb_a3,
                                 self.chkb_a4, self.chkb_a5, self.chkb_a6, self.chkb_a7 ]
        
        self.analog_progbars  = [self.prog_a0, self.prog_a1, self.prog_a2, self.prog_a3,
                                 self.prog_a4, self.prog_a5, self.prog_a6, self.prog_a7 ]
        
        self.digitalOut_checkboxes = [None,None, self.chkb_d2, None,self.chkb_d4,None,None, self.chkb_d7,
                                      self.chkb_d8, None, None, None, self.chkb_d12, self.chkb_d13]
        
        self.digitalPwm_checkboxes = [None, None, None, self.chkb_d3,None,self.chkb_d5, self.chkb_d6,
                                      None, None, self.chkb_d9, self.chkb_d10, self.chkb_d11, None, None]
        
        self.sliders_pwm = [None, None, None, self.sld_d3, None, self.sld_d5, self.sld_d6,
                            None, None, self.sld_d9, self.sld_d10, self.sld_d11, None, None ]
        
        
        #... analog check boxes        
        self.chkb_a0.stateChanged.connect(lambda: self.enable_analog(0))
        self.chkb_a1.stateChanged.connect(lambda: self.enable_analog(1))
        self.chkb_a2.stateChanged.connect(lambda: self.enable_analog(2))
        self.chkb_a3.stateChanged.connect(lambda: self.enable_analog(3))
        self.chkb_a4.stateChanged.connect(lambda: self.enable_analog(4))
        self.chkb_a5.stateChanged.connect(lambda: self.enable_analog(5))
        # self.chkb_a6.stateChanged.connect(lambda: self.enable_analog(6))
        # self.chkb_a7.stateChanged.connect(lambda: self.enable_analog(7))
        

        #... non pwm-pins check boxes
        self.chkb_d2.stateChanged.connect(lambda: self.toggle_digitalOut(2) )
        self.chkb_d4.stateChanged.connect(lambda: self.toggle_digitalOut(4) )
        self.chkb_d7.stateChanged.connect(lambda: self.toggle_digitalOut(7) )
        self.chkb_d8.stateChanged.connect(lambda: self.toggle_digitalOut(8) )
        self.chkb_d12.stateChanged.connect(lambda: self.toggle_digitalOut(12) )
        self.chkb_d13.stateChanged.connect(lambda: self.toggle_digitalOut(13) )

        
        #... pwmable-pins
        self.pwm_multiplier = [0,0,0,1/255,0,1/255,1/255,0,0,1/255,1/255,1/255,0,0]
        self.chkb_d3.stateChanged.connect( lambda: self.change_pwm_mode(3) )
        self.chkb_d5.stateChanged.connect( lambda: self.change_pwm_mode(5) )
        self.chkb_d6.stateChanged.connect( lambda: self.change_pwm_mode(6) )
        self.chkb_d9.stateChanged.connect( lambda: self.change_pwm_mode(9) )
        self.chkb_d10.stateChanged.connect( lambda: self.change_pwm_mode(10) )
        self.chkb_d11.stateChanged.connect( lambda: self.change_pwm_mode(11) )

        
        #... sliders
        self.sld_d3.valueChanged.connect( lambda: self.change_pwm_value(3))
        self.sld_d5.valueChanged.connect( lambda: self.change_pwm_value(5))
        self.sld_d6.valueChanged.connect( lambda: self.change_pwm_value(6))
        self.sld_d9.valueChanged.connect( lambda: self.change_pwm_value(9))
        self.sld_d10.valueChanged.connect( lambda: self.change_pwm_value(10))
        self.sld_d11.valueChanged.connect( lambda: self.change_pwm_value(11))

            
    def __del__(self):
            print("Goodbye!")
            try:
                self.board.shutdown()
                self.timer.stop()
            except:
                None 
          
            
    def load_comports(self):
        self.statusbar.showMessage('Loading available COM ports...')        
        self.list_comports.clear() 
        sleep(0.1)
        
        self.list_comports.addItems( find_devices() )
        self.statusbar.showMessage('Found %d com ports.' % self.list_comports.count() )
        self.pb_reload_comports.setText('Reload')

    def connect(self):
        comport = self.list_comports.currentText()
        
        # Connection instructions
        if self.board is None:
            self.statusbar.showMessage('Trying to connect through port %s...' % (comport) )
            sleep(0.1)
            
            try:
                # open board
                self.board = pymata4.Pymata4( com_port = comport, baud_rate=57600 )
                
                # define all digital pins as outputs
                _ = [ self.board.set_pin_mode_digital_output(jj) for jj in (2,4,7,8,12,13) ]
                
                # enable gui interface with arduino pins
                self.arduino_interface(True)
                
                # start timer to update analog reads
                self.timer.start( int(1000*self.timer_update_time ) )
                
                self.pb_connect.setText('Disconnect')    
                self.statusbar.showMessage('Success connecting through port %s.' % (comport) )
                return
            
            except:
                self.statusbar.showMessage('Error connecting through port %s.' % (comport) )
                return
                
        # Disconnection instructions
        if self.board is not None:
            
            try:
                # stop timer updating analog readings
                self.timer.stop() 
                
                # disable interface
                self.arduino_interface(False)
                
                # shitdown board
                self.board.shutdown()
                self.board = None
                
                self.pb_connect.setText('Connect...')
                self.statusbar.showMessage('Successfully disconnected!')
                return
            
            except:
                self.statusbar.showMessage('Error disconnecting port %s.' % (comport) )
                return
            

    def enable_analog(self, pin_number):
        #turn on
        if self.analog_checkboxes[pin_number].isChecked(): 
            # self.board.set_pin_mode_analog_input( pin_number, callback = self.update_analog, differential=3)
            self.board.set_pin_mode_analog_input( pin_number, callback = None, differential=1)
            self.statusbar.showMessage('Enabling reporting on analog pin A%d.' % pin_number )
        
        #turn off
        else:
            self.board.disable_analog_reporting(pin_number)
            self.statusbar.showMessage('Disabling reporting on analog pin A%d.' % pin_number )

    # def update_analog(self,data):
    #     self.analog_progbars[ data[CB_PIN] ].setValue( data[CB_VALUE] )
        

    def toggle_digitalOut(self, pin_number):        
        value = self.digitalOut_checkboxes[pin_number].checkState()/2

        try:
            self.board.digital_write(pin_number, value)
            self.statusbar.showMessage('Pin D%d is now %s'% (pin_number , 'HIGH' if value else 'LOW') )
        except:
            self.statusbar.showMessage('Error changing pin D%d state to %s' % (pin_number , 'HIGH' if value else 'LOW') )
            
    def change_pwm_mode(self, pin_number):
        box_state = self.digitalPwm_checkboxes[pin_number].checkState()
            
        if box_state == 0:
            self.board.set_pin_mode_digital_output(pin_number)
            self.pwm_multiplier[pin_number] = 1/255
            self.statusbar.showMessage('Pin D%d is now in OUTPUT mode' % pin_number)
        elif box_state==1:
            self.board.set_pin_mode_pwm_output(pin_number)
            self.pwm_multiplier[pin_number] = 1
            self.statusbar.showMessage('Pin D%d is now in PWM mode' % pin_number)
        elif box_state==2:
            self.board.set_pin_mode_servo(pin_number, min_pulse=544, max_pulse=2400)
            self.pwm_multiplier[pin_number] = 180/255
            self.statusbar.showMessage('Pin D%d is now in SERVO mode' % pin_number)
        

    def change_pwm_value(self, pin_number):
        box_state = self.digitalPwm_checkboxes[pin_number].checkState()
        value = int( self.sliders_pwm[pin_number].value() * self.pwm_multiplier[pin_number] )
        try:
            if box_state == 0:
                self.board.digital_write(pin_number, value )
            elif box_state ==1:
                self.board.pwm_write(pin_number, value)
            elif box_state == 2:
                self.board.servo_write(pin_number, value)
            self.statusbar.showMessage('Slider of D%d has now a value of %d.' % (pin_number, value) )
        
        except:
            self.statusbar.showMessage('Error changing digital output of D%d.' % pin_number )


    def update_analog_values(self):
        try:
            self.new_data_vector = np.zeros((8,1))
            if self.chkb_a0.isChecked(): self.__update_analog_values(0) 
            if self.chkb_a1.isChecked(): self.__update_analog_values(1) 
            if self.chkb_a2.isChecked(): self.__update_analog_values(2) 
            if self.chkb_a3.isChecked(): self.__update_analog_values(3) 
            if self.chkb_a4.isChecked(): self.__update_analog_values(4) 
            if self.chkb_a5.isChecked(): self.__update_analog_values(5)
            self.__update_historic_data()

        except:
            self.timer.stop()
            self.statusbar.showMessage('Error updating analog values!')
           
    
    def __update_analog_values(self, pin_number):
        value =  int( self.board.analog_read(pin_number)[0] )
        self.analog_progbars[pin_number].setValue( value )
        self.new_data_vector[pin_number] = value
        
    def __update_historic_data(self):
        # update analog values 
        self.historic_data = np.append( self.historic_data, self.new_data_vector, axis=1)      
        if self.historic_data.shape[1] > self.max_historic_data:
            self.historic_data = self.historic_data[:,1:]
        
        # update time vector
        if self.time_vector.shape[0] < self.max_historic_data:
               self.time_vector   = np.append( self.time_vector-self.timer_update_time  , np.zeros((1,1)) )      
        
        # update analog plots
        self.plot_canvas1.axes.cla()  # Clear the canvas.
        self.plot_canvas1.axes.plot( self.time_vector, self.historic_data.T )
        self.plot_canvas1.axes.grid()
        self.plot_canvas1.axes.set_xlabel('Time (s)')
        self.plot_canvas1.axes.set_ylabel('Analog values')
        self.plot_canvas1.axes.legend(('A0','A1','A2','A3','A4','A5'), bbox_to_anchor=(0.85, 0.1, 0.5, 1) )

        self.plot_canvas1.draw()
        

        
    def arduino_interface(self, value):
        self.prog_a0.setEnabled(value)
        self.prog_a1.setEnabled(value)
        self.prog_a2.setEnabled(value)
        self.prog_a3.setEnabled(value)
        self.prog_a4.setEnabled(value)
        self.prog_a5.setEnabled(value)
        self.prog_a6.setEnabled(value)
        self.prog_a7.setEnabled(value)
        
        self.chkb_a0.setEnabled(value)
        self.chkb_a1.setEnabled(value)
        self.chkb_a2.setEnabled(value)
        self.chkb_a3.setEnabled(value)
        self.chkb_a4.setEnabled(value)
        self.chkb_a5.setEnabled(value)
        self.chkb_a6.setEnabled(value)
        self.chkb_a7.setEnabled(value)

        self.chkb_d2.setEnabled(value)
        self.chkb_d3.setEnabled(value)
        self.chkb_d4.setEnabled(value)
        self.chkb_d5.setEnabled(value)
        self.chkb_d6.setEnabled(value)
        self.chkb_d7.setEnabled(value)
        self.chkb_d8.setEnabled(value)
        self.chkb_d9.setEnabled(value)
        self.chkb_d10.setEnabled(value)
        self.chkb_d11.setEnabled(value)
        self.chkb_d12.setEnabled(value)
        self.chkb_d13.setEnabled(value)
    
        self.sld_d3.setEnabled(value)
        self.sld_d5.setEnabled(value)
        self.sld_d6.setEnabled(value)
        self.sld_d9.setEnabled(value)
        self.sld_d10.setEnabled(value)
        self.sld_d11.setEnabled(value)
        
        self.list_comports.setEnabled(not value)
        self.pb_reload_comports.setEnabled(not value)
                
        
app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application













