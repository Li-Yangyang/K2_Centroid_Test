# K2_Centroid_Test
This is a prime python code for conducting centroid test for K2/Kepler target. This code origins from pipeline desctription from [Kepler Data Processing Handbook](https://archive.stsci.edu/kepler/manuals/KSCI-19081-002-KDPH.pdf)
## Dependencies
- numpy
- scipy
- matplotlib
- astropy
## Usage
1. Clone or download this code from github
2. Main function is in PRFfunc.py. 
3. Modify the stellar parameters as model.ini shows
4. Input the light curve file and pixel fits file, here I use light curve from EVEREST pipeline and pixel fits from K2 archive from MAST
5. When running this code, you can derive two figures about out-of-transit image and difference image and one figure about dignosis indicating whether the transiting is caused by planet or not. 

