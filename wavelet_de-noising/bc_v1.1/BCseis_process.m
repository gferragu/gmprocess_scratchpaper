function sacdata_out=BCseis_process(sacdata_in,params)
% BCseis_process - perform processing steps on data as determined by
% initial analysis with the BCseis GUI
% version 1.1, March 7, 2019

% Input:
%       sacdata_in - waveform in SAC format
%       params - data structure containing processing information
%
% Output:
%       sacdata_out - processed waveform in SAC format

%****** params data structure *********************************************
% params.wavelet_type = "morlet" / "shannon" / "mhat" / "hhat"
% params.nvoices (4 8 16 32)
% params.nbpblck = 0, no bandpass block applied; = 1 bandpass block applied
% params.scale_min = minimum scale for bandpass block
% params.scale_max = maximum scale for bandpass block
% params.bthresh = percent amplitude for the block
% params.nnbnd = 0, no noise model considered; =1, compute a noise model
% params.tstrn = noise start time in seismogram
% params.tfinn = noise finish time in seismogram
% params.noisethresh = 0, do not apply non-linear threshold to the noise
%                    = 1, apply hard threshold
%                    = 2, apply soft threshold
% params.signalthresh = 0, do not apply non-linear threshold to the signal
%                     = 1, apply hard threshold
%                     = 2, apply soft threshold
% params.nsig = number of standard deviations for the block threshold
%             = -1, apply Donoho's Threshold criterion
%             = -2, apply Empiridal CDF method to determine threshold
%                  (recommended)
% params.nsnr = 0, do not apply the SNR detection method;
%             = 1, apply.  If applied, it will be completed before hard
%                  thresholding.
% params.nsnrlb = percent lower bound for SNR detection method
%
% Note: there is no automatic way provided to perform the "include" or
% "exclude" Interactive Block Choice function from the GUI.
%**************************************************************************
% compute the CWT
sacdata_new=sacdata_in;
sacdata_out=sacdata_new;

delta=1.0./sacdata_new(1).samprate;
npts=sacdata_new(1).nsamps;
x=sacdata_new(1).data(:,1);
t=linspace(0,delta.*(npts-1),npts);

wavelet_type=params.wavelet_type;
nvoices=params.nvoices;

[Wx_new,as_new] = cwt_fw(x,wavelet_type,nvoices,delta);

[na,n] = size(Wx_new);

%**************************************************************************
% Apply a block bandpass, if wanted
if params.nbpblck == 1
        %  Gather parameters
    scale_min=params.scale_min;
    scale_max=params.scale_max;
    thresh=params.bthresh.*0.01;
    
    %  save old CWT
    Wx_old=Wx_new;
    as_old=as_new;
    
    %  Threshold Wx_old to get Wx_new
    a(1,1:na)=1.0;
    a=a.*(as_old <= scale_min | as_old >= scale_max);
    for k=1:na
        if a(k) == 0
            a(k)=thresh;
        end
    end
    Wx_new=Wx_old.*a';
else
end

%**************************************************************************
% Calculate noise model and threshold function, if needed
if (params.nnbnd == 1 || params.noisethresh > 0 || params.signalthresh > 0)
    % Get the time window

    begtime=params.tstrn;
    endtime=params.tfinn;

    nbeg=round(begtime./delta) + 1;
    nend=round(endtime./delta) + 1;
    
    if params.nsig == -1
        % compute Donoho's Threshold Criterion
        params.nsig=sqrt(2.*log10(nend-nbeg +1));
    end

    if params.nsig > -2
        % Assume Gaussian statistics
        % Calculate the mean and standard deviations at each scale in the
        % time-scale window
        M=mean(abs(Wx_new(:,nbeg:nend)'));
        S=std(abs(Wx_new(:,nbeg:nend)'));
        P=M + params.nsig.*S;
    else
        % Estimate empirical CDF for each scale, calculate 99% confidence 
        % level for the noise threshold
        [nrow,ncol]=size(Wx_new);
        conf=0.99;
        n_noise=nend-nbeg+1;
        for k=1:nrow
            W(1:n_noise)=abs(Wx_new(k,nbeg:nend))';
            [f,x]=ecdf(W);
            P(k)=interp1(f,x,conf);
        end
    end

else
end

%**************************************************************************
% apply the SNR detection method if wanted
if (params.nnbnd == 1 && params.nsnr == 1)
    
    nlbound=params.nsnrlb.*0.01;
    M_max=max(abs(M));
    Wx_new=Wx_new./(M'+nlbound.*M_max);
    
    % recalculate the noise model for possible further use
    M=mean(abs(Wx_new(:,nbeg:nend)'));
    S=std(abs(Wx_new(:,nbeg:nend)'));
else
end

%**************************************************************************
% Apply hard thresholding to the noise, if wanted (removing noise)
if params.noisethresh == 1
    Wx_old=Wx_new;
    W_test=abs(Wx_old);
    Wx_new=Wx_old.*(P' < W_test);
else
end

%**************************************************************************
% Apply soft thresholding to the noise, if wanted (removing noise)
if params.noisethresh == 2
    Wx_old=Wx_new;
    W_test=abs(Wx_old);
    Wx_new=sign(Wx_old).*(W_test - P').*(P' < W_test);
else
end

%**************************************************************************
% Apply hard thresholding to the signal, if wanted (removing signal)
if params.signalthresh == 1
    Wx_old=Wx_new;
    W_test=abs(Wx_old);
    Wx_new=Wx_old.*(P' > W_test);
else
end

%**************************************************************************
% Apply soft thresholding to the signal, if wanted (removing signal)
if params.signalthresh == 2
    Wx_old=Wx_new;
    W_test=abs(Wx_old);
    Wx_new=sign(Wx_old).* P'.*(P' <= W_test) + W_test.*(P' > W_test);
else
end

%**************************************************************************
% Compute the inverse CWT
anew=cwt_iw(Wx_new,wavelet_type,nvoices);
sacdata_out(1).data(1:npts,1)=anew(1:npts);

end

