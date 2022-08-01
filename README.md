# EDS2CHEM
EDS2CHEM is a Bayesian-informed Markov-Chain Monte Carlo code applied to determine best-fit profile locations and scaling factors to convert measured EDS peak intensities to composition.

## Code 
- MCMC provides the pixel projection and MC$^3$ code for determining best-fit profile locations and scaling factors. 
- EDS2CHEM_Example showcases how one instance of the code can be run for one analyzed crystal. 
- MC3_Plotting, QEMSCAN_functions, QEMSCAN_plotting are backend plotting functions. 
- PixelwiseAnConversion is another MCMC run to determine mean XAn compositions for individual pixels after determining best-fit scaling factors for all profiles in one thin section or mount. 

## Data
Data can be found here: https://drive.google.com/drive/folders/1jvN2R5M4f0a8Rou7iPy1rKIFndRNmmLo?usp=sharing