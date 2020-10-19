function BCseis_process_driver
% BCseis_process_driver - program to setup and run BCseis_process for
% applying block manipulation of a signal's CWT as determined from a
% session with BCseis.
%
% Determine appropriate processes and parameters using the BCseis GUI.
% Then set the wanted parameters here to process many seismograms as, for
% example, from an array.

% C. A. Langston, July 9, 2018
% version 1.1, March 7, 2019

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

% Set the params data structure for this run
params.wavelet_type = "morlet";
params.nvoices=16;
params.nbpblck = 1;
params.scale_min = 1.0;
params.scale_max = 200.0;
params.bthresh=0.0;
params.nnbnd = 1;
params.tstrn = 0.0;
params.tfinn = 60.0;
params.noisethresh = 2;
params.signalthresh = 0;
params.nsig = -2.0;
params.nsnr = 0;
params.nsnrlb = 1.0;

% set the filename suffix for this run
fsuffix='stbr1';

% Read a file of SAC filenames
%   Call uigetfile menu

[filename,dirname]=uigetfile('*.*','Pick a File of SAC filenames for processing');
if filename == 0; return; end;

sacnamefile=strcat(dirname,filename);

%   Read in sacfiles from sacnamefile
fid=fopen(sacnamefile,'r');

j=1; while feof(fid)==0;
    filedum=[];
    filedum2=[];
    filedum=fgetl(fid);
    [filedum2,fcount,errmsg,nextindex]=sscanf(filedum,'%s',80);
    file(j,1:nextindex-1)=filedum2(1:nextindex-1);
    j=j+1;
    end;  
[nfiles,dull]=size(file);
fclose(fid);

%   Use DOE MatSeis readsac, 'sacdata' is a structure with all the data.
endian='l';
sacdata_in=readsac2( file, endian );

% initialize sacdata_out
sacdata_out=sacdata_in;

% process each file

h=waitbar(0,'Processing Files');
for k=1:nfiles
    sacdata_test=BCseis_process(sacdata_in(k),params);
    
    % add the suffix to the filename for later writing
    filename_old=sacdata_out(k).filename;
    sacdata_out(k)=sacdata_test(1);
    sacdata_out(k).filename=strcat(filename_old,'.',fsuffix);
    
    waitbar(k / nfiles);
end

close(h);

% write out all files
writesac_a(nfiles,sacdata_out);

% All Done

end

