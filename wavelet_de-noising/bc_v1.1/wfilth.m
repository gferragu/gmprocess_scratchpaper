function [psih] = wfilth(type, N, a, opt)
% Outputs the FFT of the wavelet of family 'type' with parameters
% in 'opt', of length N at scale a: (psi(-t/a))^.
%
% [Inputs]
% type: wavelet type
% N: number of samples to calculate
% a: wavelet scale parameter
% opt: wavelet options
% opt.dt: delta t
%
% [Outputs]
% psih: wavelet sampling in frequency domain
%---------------------------------------------------------------------------------
opt = struct();
k = 0:(N-1);
xi = zeros(1, N);
xi(1:N/2+1) = 2*pi/N*[0:N/2];
xi(N/2+2:end) = 2*pi/N*[-N/2+1:-1];
psihfn = wfiltfn(type, opt);
psih = psihfn(a*xi);
% Normalizing
psih = psih * sqrt(a) / sqrt(2*pi);
% Center around zero in the time domain
psih = psih .* (-1).^k;
end

