February 28th 2021 - Gabe Ferragut

#########################################################################################
The /scripts folder contains a few directories 
with generally different purposes or main functions

#########################################################################################

/scripts

	/noise

		- The meat and potatoes modules:

			- noise_generation.py
			
				- Messy area to house and test noise dataset building
				  functions, i/o, noise addition functions, etc.

				- Starts with the original attempts at very ideal noise
				  using gen_fft_noise() and gen_band_limited_noise()

			- compare.py
			
				- Standalone module to house functions for comparing 
				  noise models, methods, and noisy vs. denoised data.
	
				- Most of these things are transferred here from the 
				  scratch work at the bottom off models.py

				- Compares things like FFTs, spectrograms, PPSD, etc.

				- Not totally updated from models.py work

			- models.py

				- There's a lot (too much) going on in here

				- Unit conversions (dB, Hz, Period, lin and log scales)

				- Noise model i/o, interpolation, static plotting

				- Freq. domain noise signal construction

				- Boore method of stochastic signals modulated by
				  the noise models

				- Many of the functions have commented out plotting
				  functions to check values as the code progresses
				  or as noise is being generated

				- There's a large "scratchwork" area at the bottom

					- Easy area to test out plotting approaches
					  or play with the noise/data being made

					- Periodically I clean this up and add the 
					  plotting routines to compare.py
			
		- /data

			- Contains noise traces (in either miniseed or as a csv)

			- Noise comes in different "flavors"
			
				- The attempts at using purely the NHNM/NLNM

				- The attempts at the Boore method of creating noise

				- Purely stochastic random starting signals

		- /noise_models

			- Contains the NHNM and NLNM of Peterson (1993) in various forms

			- These are read in and used to calculate the noise models
		

	/synthetics
	
		- This module contains the functions and i/o for the synthetic files
		  provided by Kyle

		- There is also a /data folder that is intended to hold synthetic signals
		  that have been processed in different ways

			- /miniseed

				- /processed - bandpassed and wavelet denoised synthetics

				- /noisy - synthetics with artificial noise added

				- /clean - the original unprocessed synthetics

	/denoise

		- This is the module containing wavelet denoising functions as well as a
		  skeleton layout of the as yet unimplemented continuous wavelet transform 
		  denoising methods from the MATLAB package bc_seis by Chuck Langston
		
		- The discrete versions are implemented based on Brad Aagard's codes

		- "dwt.py" contains the actual denoising functions
		
		- "utils.py" contains some helper functions to facilitate the denoising


	/wavelet_denoising
		
		- Unused right now, was initial attempt to port some of the MATLAB code from
		  Chuck Langston's package

	/gmprocess_tests
		
		- Just a space to test out reported issues or try aspects of the codes

	/gmprocess_tutorials

		- A few short examples of using gmprocess from the documentation