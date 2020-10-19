function [Wx,as] = cwt_fw(x, type, nv, dt)
% Forward continuous wavelet transform, discretized, as described
% in Mallat, S., Wavelet Tour of Signal Processing 3rd ed.Sec. 4.3.3.
%
% [INPUTS]
% x: input signal vector.
% type: wavelet type, string
% nv: number of voices
% dt: sampling period
% opt: options structure
%
% [OUTPUTS]
% Wx: [na x n] size matrix (rows = scales, cols = times)
% as: na length vector containing the associated scales
%
%---------------------------------------------------------------------------------
% Modified after a wavelet transform by Eugene Brevdo
%---------------------------------------------------------------------------------
opt = struct();
opt.rpadded = 0;
x = x(:); % Turn into column vector
n = length(x);
% Padding the signal
N = 2^(1+round(log2(length(x)+eps)));
n1 = floor((N-n)/2);
n2 = n1;if (mod(2*n1+n,2)==1), n2 = n1 + 1; end
xl = padarray(x(:), n1, 'pre');
xr = padarray(x(:), n2, 'post');
x = [xl(1:n1); x(:); xr(end-n2+1:end)];
% Choosing more than this means the wavelet window becomes too short
noct = log2(N)-1;
assert(noct > 0 && mod(noct,1) == 0);
assert(nv>0 && mod(nv,1)==0);
assert(dt>0);
assert(~any(isnan(x)));
na = noct*nv;
as = 2^(1/nv) .^ (1:1:na);
Wx = zeros(na, N);
x = x(:).';
xh = fft(x);
% for each octave
for ai = 1:na
    a = as(ai);
    psih = wfilth(type, N, a, opt);
    xcpsi = ifftshift(ifft(psih .* xh));
    Wx(ai, :) = xcpsi;
end
% Shorten W to proper size (remove padding)
if (~opt.rpadded)
    Wx = Wx(:, n1+1:n1+n);
end
% Output a for graphing purposes, scale by dt
as = as * dt;
end
