function [psihfn] = wfiltfn(type, opt)
% Wavelet transform function of the wavelet filter in question,
% fourier domain.
%
% [Input]
% type: string (see below)
% opt: options structure, e.g. struct('s',1/6,'mu',2)
%
% [Output]
% psihfn: mother wavelet function ( mexican hat, morlet, shannon, or hermitian)
% Example:
% psihfn = wfiltfn('bump', struct('mu',1,'s',.5));
% plot(psihfn(-5:.01:5));
%---------------------------------------------------------------------------------
switch type
case 'mhat' % mexican hat
    if ~isfield(opt,'s'), s = 1; else s = opt.s; end
    psihfn = @(w) -sqrt(8)*s^(5/2)*pi^(1/4)/sqrt(3)*w.^2.*exp(-s^2*w.^2/2);
case 'morlet'
    % can be used with synsq for large enough s (e.g. >5)
    if ~isfield(opt,'mu'), mu = 2*pi; else mu = opt.mu; end
    cs = (1+exp(-mu^2)-2*exp(-3/4*mu^2)).^(-1/2);
    ks = exp(-1/2*mu^2);
    psihfn = @(w)cs*pi^(-1/4)*(exp(-1/2*(mu-w).^2)-ks*exp(-1/2*w.^2));
case 'shannon'
    psihfn = @(w)exp(-1i*w/2).*(abs(w)>=pi & abs(w)<=2*pi);
case 'hhat' % hermitian hat
    psihfn = @(w)2/sqrt(5)*pi^(-1/4)*w.*(1+w).*exp(-1/2*w.^2);
% case 'mostafa',
    % load ss
    % if ~isfield(opt,'mu'), mu = 5; else mu = opt.mu; end
    % if ~isfield(opt,'s'), s = 1; else s = opt.s; end
    % psihfnorig = @(w)(0.0720*w.^8)+(0.2746*w.^7)+(0.2225*w.^6)+(-0.2781*w.^5)+(-0.3884*w.^4)+(0.0735*w.^3)+(-0.3354*w.^2)+(-0.0043*w)+(0.3675);
    % psihfn = @(w) psihfnorig((w-mu)/s);
otherwise
error('Unknown wavelet type: %s', type);
end
