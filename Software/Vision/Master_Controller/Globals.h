#ifndef _GLOBALS_H    
#define _GLOBALS_H

#define POT_MIN 0
#define POT_MAX 1023

//Axis 1 is currently calibrated for 1000 microsteps. For 180 degrees of rotation, we will only allow 500 steps.
#define AXIS_1_RANGE_MIN 0 
#define AXIS_1_RANGE_MAX 500
#define AXIS_1_GEAR_RATIO 6

//EMA Parameters
const float EMA_ALPHA = 0.05;

//Stepper control
#define DEADZONE 0 //Prevents stepper moving due to samll noise on the Potentiometer input

#endif
