#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:29:16 2020

@author: gabriel
"""
import math
import numpy as np

#%% MATLAB Code to reproduce

# function x = cwt_iw(Wx, type, nv)
# % The inverse wavelet transform
# %
# % Implements Eq. (4.67) of Mallat, S., Wavelet Tour of Signal Processing 3rd ed.
# %
# % Inputs:
# % Wx: wavelet transform of a signal, see help cwt_fw
# % type: wavelet used to take the wavelet transform,
# % see help cwt_fw and help wfiltfn
# % opt: options structure used for forward wavelet transform.
# %
# % Output:
# % x: the signal, as reconstructed from Wx
# %
# %---------------------------------------------------------------------------------
# % Modified after a wavelet transform written by Eugene Brevdo
# %---------------------------------------------------------------------------------
# opt = struct();
# [na, n] = size(Wx);
# % Padding the signal
# N = 2^(1+round(log2(n+eps)));     ### eps in matlab is for floating point rel. accuracy...
# n1 = floor((N-n)/2);
# n2 = n1;if (mod(2*n1+n,2)==1), n2 = n1 + 1; end
# Wxp = zeros(na, N);
# Wxp(:, n1+1:n1+n) = Wx;
# Wx = Wxp; clear Wxp;
# noct = log2(N)-1;
# as = 2^(1/nv) .^ (1:1:na);
# assert(mod(noct,1) == 0);
# assert(nv>0 && mod(nv,1)==0);
# % the admissibility coefficient Cpsi
# switch type
# case 'shannon',
# Cpsi = log(2);
# otherwise
# psihfn = wfiltfn(type, opt);
# Cpsi = quadgk(@(x) (conj(psihfn(x)).*psihfn(x))./x, 0, Inf);
# end
# % Normalize
# Cpsi = Cpsi / (4*pi);
# x = zeros(1, N);
# for ai=1:na
# a = as(ai);
# Wxa = Wx(ai, :);
# psih = wfilth(type, N, a, opt);
# % Convolution theorem
# Wxah = fft(Wxa);
# xah = Wxah .* psih;
# xa = ifftshift(ifft(xah));
# x = x + xa/a;
# end
# % Take real part and normalize by log_e(a)/Cpsi
# x = log(2^(1/nv))/Cpsi * real(x);
# % Keep the unpadded part
# x = x(n1+1: n1+n);
# end


#%% Python Implementation 

def cwt_iw(Wx, type, nv):
    
    #Instead of using an "options" structure like in Chuck's code, we'll set up default and optional args I think
    
        ### But #### options is never passed as an arg ... it's just defined in the functions?
    
    # opt = options     # Options inclucde: rpadded, dt, mu, s
    

    ## Need to get the size of the Wavelet signal. In MATLAB, this is a matrix ... Here probably use two arrays.
    
    # Get the length of the input time series for octave/voice calculation
    na = len(Wx)
    n = len(Wx[:][0])
    
    # Padding the signal (optionally)
        # What exactly is big N in Chuck's code?
        
    N = 2**(1 + round(math.log2(n)))
    
    n1 = (N-n)/2     #Possibly apply Python version of matlab floor() function here
    n2 = n1
    
    if (2*n1+n) % 2 == 1:
        n2 = n1 + 1
    
    # How Chuck does it:
    Wxp = np.zeros(na, N)
    Wx = Wxp[:][(n+1):(n1+n)]
    Wx = Wxp;
    
    ### This makes zero sense to me ###
    
    
    # Simpler padding, but first: ### WHy exactly is the padding needed? ###
    
    
    return