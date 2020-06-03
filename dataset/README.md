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

2. File 'advertisement_bandits.csv': 
------------------------------------
It indicates the bandits who are advertised or not after breaking the law.
Contents:
- advertisement_id: bandit ID 
- action: a boolean indicating if a certain bandit is or not advertised
