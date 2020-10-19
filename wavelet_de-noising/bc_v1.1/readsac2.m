function s = readsac2 ( filenames, endian )
%READSAC	Read SAC files.
%	SAC = READSAC ( FILENAMES ) reads the given SAC files. FILENAMES may 
%	be a string, string matrix, or string cell array. The data is 
%	returned in a structure SAC. 
%
%	READSAC will only read evenly-spaced time series data files. The 
%	'status' field should be checked to verify that the file was opened 
%	without errors and the file contained evenly-spaced time series data. 
%	If status = 0, only the 'filename' field will contain valid data. 
%	Fields that are set to undefined in the SAC file will be NaN or '' 
%	in the SAC structure. 
%
%	The SAC structure is defined to have these fields: 
%	    status   - Status of structure (ok=1, error=0).
%	    filename - File name of SAC file.
%	    sacver   - SAC header version number.
%	    loaddate - Load date (epochal).
%	    station  - Station name.
%	    channel  - Channel name.
%	    network  - Name of seismic network.
%	    region   - Geographic region of station.
%	    location - Station location [ lat, lon, elev, depth ].
%	    insttype - Instrument type.
%	    response - Instrument response.
%	    cmpname  - Component name.
%	    cmpaz    - Component azimuth.
%	    cmpinc   - Component incident angle.
%	    time     - Data start time.
%	    samprate - Data sample rate.
%	    nsamps   - Number of samples.
%	    scale    - Scale factor.
%	    sigtype  - Signal type (displacement, velocity, acceleration).
%	    quality  - Signal quality ('good', glitches, dropouts, low snr).
%	    evname   - Event name.
%	    evid     - Event ID.
%	    evloc    - Event location [ lat, lon, depth,  elev].
%	    evtime   - Event time.
%	    evend    - Event end time.
%	    evendid  - Event end ID.
%	    evtype   - Event type (nuclear, earthquake, etc.).
%	    evregion - Geographic region of event.
%	    evdist   - Station to event distance (deg).
%	    evaz     - Event to station azimuth.
%	    evbaz    - Station to event azimuth.
%	    evdistkm - Station to event distance (km).
%	    artime   - Arrival times (column vector, up to 11 arrivals).
%	    arid     - Arrival IDs (string matrix).
%	    user     - User defined values (10 row column vector).
%	    kuser    - User defined values (3 row string cell array).
%	    data     - Time series data (column vector).
%

%	MatSeis 1.6
%	Mark Harris, mharris@sandia.gov
%	Copyright (c) 1996-2001 Sandia National Laboratories. All rights reserved.

%   Modified 7//04/05 C.A. Langston to include choice of little/big endian
%   read.  Also modified sac structure to save A, B, and C header arrays to
%   make it easier to write files later.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default output.
%
s = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check arguments.
%
if nargin<1
  return;
end
if isempty(filenames);
  return;
end
files = cellstr(filenames);
nfiles = size(files,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize output structure.
%
s = sac( nfiles );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read all files.
%
for f=1:nfiles;
  s(f).status   = logical(0);
  s(f).filename = char(files(f));

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Open SAC file.
  %
  fid = fopen( char(files(f)), 'r', endian);
  if fid ~= -1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Read SAC file header.
    %
    A = fread( fid, [ 70 1 ], 'float32' );
    B = fread( fid, [ 40 1 ], 'int32' );
    C = char( fread ( fid, [ 1 192 ], 'char' ) );
    %size(C);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save the Raw input header for ease of
    % writing it out
    %
    % C.A. Langston 6/29/05
    %
    s(f).headerA = A;
    s(f).headerB = B;
    s(f).headerC = C;
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Read SAC file data.
    %
    D = fread( fid, 'float32' );

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set undefined values.
    %
    A(A==-12345.0) = NaN;
    B(B==-12345) = NaN;
    C = cellstr(reshape(C,8,24)');
    C(strmatch('-12345',C)) = {''};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Close SAC file.
    %
    fclose( fid );

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check content of SAC file.
    % Only evenly-spaced time series are supported.
    %
    if ( B(16) == 1 | isnan(B(16)) ) & ( B(36) == 1 | isnan(B(36)) )

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Calculate SAC reference time.
      %
      
      %  initialize time if not defined
      if isnan(B(1)); B(1)=1970; end;
      if isnan(B(2)); B(2)=1; end;
      if isnan(B(3)); B(3)=0; end;
      if isnan(B(4)); B(4)=0; end;
      if isnan(B(5)); B(5)=0; end;
      if isnan(B(6)); B(6)=0; end;
      
      if B(1) < 100
        B(1) = B(1) + 1900;
      end
      [y,m,d] = ymd(B(1)*1000+B(2));
      reftime = htoe( [ y m d B(3) B(4) (B(5)+B(6)/1000) ] );

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Fill in output structure.
      %
      s(f).status   = logical(1);
      s(f).sacver   = B(7);
      if ~isempty(C{23})
        s(f).loaddate = htoe( str2time ( C{23} ) );
      end
      

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Get station information.
      %
      s(f).station  = C{1};
      s(f).channel  = C{21};

      switch B(17)
      case 6,    sigtype = 'displacement';
      case 7,    sigtype = 'velocity';
      case 8,    sigtype = 'acceleration';
      case 50,   sigtype = 'volts';
      case 5,    sigtype = 'unknown';
      otherwise, sigtype = '';
      end

      switch B(24)
      case 45,   quality = 'good';
      case 46,   quality = 'glitches';
      case 47,   quality = 'dropouts';
      case 48,   quality = 'lowsnr';
      case 44,   quality = 'other';
      otherwise, quality = '';
      end

      s(f).network  = C{22};
      s(f).region   = B(21);
      s(f).location = A(32:35)';
      s(f).insttype = C{24};
      s(f).response = A(22:31)';
      s(f).cmpname  = C{21};
      s(f).cmpaz    = A(58);
      s(f).cmpinc   = A(59);
      s(f).time     = reftime + A(6);
      s(f).samprate = 1/A(1);
      s(f).nsamps   = B(10);
      s(f).scale    = A(4);
      s(f).sigtype  = sigtype;
      s(f).quality  = quality;

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Get event information.
      %
      switch B(23)
      case 37,   evtype = 'nuclear';
      case 38,   evtype = 'nuclear preshot';
      case 39,   evtype = 'nuclear postshot';
      case 40,   evtype = 'earthquake';
      case 41,   evtype = 'foreshock';
      case 42,   evtype = 'aftershock';
      case 43,   evtype = 'explosion';
      case 44,   evtype = 'other';
      otherwise, evtype = '';
      end

      s(f).evname   = [C{2:3}];
      s(f).evid     = C{5};
      s(f).evloc    = [ A(36) A(37) A(39) A(38) ];
      s(f).evtime   = reftime + A(8);
      s(f).evend    = reftime + A(21);
      s(f).evendid  = C{17};
      s(f).evtype   = evtype;
      s(f).evregion = B(22);
      s(f).evdist   = A(54);
      s(f).evaz     = A(52);
      s(f).evbaz    = A(53);
      s(f).evdistkm = A(51);

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Get arrival information.
      %
      arid = C(6:16);
      artime = reftime + [ A(9); A(11:20) ];
      for a = 1:11
        if ~isempty(arid{a})
          s(f).arid     = strvcat ( s(f).arid, arid{a} );
          s(f).artime   = [ s(f).artime; artime(a) ];
        end
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Get user values.
      %
      s(f).user = A(41:50);
      s(f).kuser = C(18:20);

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Get data vector.
      %
      s(f).data = D;

    else
      disp ( [ 'Warning: SAC file ''' char(files(f)) ''' is not an even time series.' ] );

    end

  else
    disp ( [ 'Warning: Could not open file ''' char(files(f)) '''.' ] );

  end

end

function s = sac ( n )
%SAC	Initialize a SAC structure.
%	SAC ( N ) returns an [Nx1] empty SAC structure. N defaults to 1. 
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check arguments.
%
if ( nargin < 1 )
  n = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize output structure.
%
%  modified 6/29/05 C.A. Langston to include
%  raw input SAC header values

sacfields = ...
[ ...
  'headerA '; ...
  'headerB '; ...
  'headerC '; ...
  'status  '; ...
  'filename'; ...
  'sacver  '; ...
  'loaddate'; ...
  'station '; ...
  'channel '; ...
  'network '; ...
  'region  '; ...
  'location'; ...
  'insttype'; ...
  'response'; ...
  'cmpname '; ...
  'cmpaz   '; ...
  'cmpinc  '; ...
  'time    '; ...
  'samprate'; ...
  'nsamps  '; ...
  'scale   '; ...
  'sigtype '; ...
  'quality '; ...
  'evname  '; ...
  'evid    '; ...
  'evloc   '; ...
  'evtime  '; ...
  'evend   '; ...
  'evendid '; ...
  'evtype  '; ...
  'evregion'; ...
  'evdist  '; ...
  'evaz    '; ...
  'evbaz   '; ...
  'evdistkm'; ...
  'artime  '; ...
  'arid    '; ...
  'user    '; ...
  'kuser   '; ...
  'data    '; ...
];

s = cell2struct ( cell(size(sacfields,1),n), cellstr(sacfields), 1 );

