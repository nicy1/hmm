DATA SET FILES DESCRIPTION
--------------------------

1. File 'tele_electrocardiogram.csv': 
-------------------------------------
It contains the measurements of electrical activity of patients heart to show whether or not it is working normally. 
An ECG records the heart's rhythm and activity on a moving strip of paper or a line on a screen.
Contents:
- ECG_value: the ECG sample value (mV)
- QRS_location: a bolean indicating the locations of annotated qrs complexes
- visually_mask: a boolean indicating the visaully determined mask
- software_mask: a boolean indicating the software determined mask

2. File 'touch_dynamics.csv': 
-----------------------------
It contains touch dynamic informations of participants, collected using TouchSense (available at 'https://play.google.com/store/apps/details?id=org.mun.navid.touchsens'). While the user interacts with the keyboard, it captures the touch inputs corresponding to thodse actions and stores them in a data file.
Contents:
- pressure_numeric: a pressure applied by a touch action
- size_numeric: number of pixels affected on the screen by a touch action
- touchmajor_numeric: major axis of an ellipse that represented the touched area
- touchminor_numeric: minor axis of an ellipse that represented the touched area
- duration_numeric: time interval from the moment a finger touches the screen until the finger loses contact with it
- flytime_numeric: shows the time elapsed between finishing typing a character and starting to type the next one
- shake_numeric: amount of vibration of the smartphone while performing touch actions
- orientation_numeric: recors whether the touch behavior was recorded while the device is in the landscape orientation or the portrait one
- type_numeric: recors whether the touch behavior typing in a word or in a number
- class: android id /others

3. File 'advertisement_bandits.csv': 
------------------------------------
It indicates the bandits who are advertised or not after breaking the law.
Contents:
- advertisement_id: bandit ID 
- action: a boolean indicating if a certain bandit is or not advertised
