function varargout = BCseis(varargin)
% BCSEIS MATLAB code for BCseis.fig
%      BCSEIS, by itself, creates a new BCSEIS or raises the existing
%      singleton*.
%
%      H = BCSEIS returns the handle to a new BCSEIS or the handle to
%      the existing singleton*.
%
%      BCSEIS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BCSEIS.M with the given input arguments.
%
%      BCSEIS('Property','Value',...) creates a new BCSEIS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BCseis_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BCseis_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BCseis

% Last Modified by GUIDE v2.5 06-Mar-2019 07:59:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BCseis_OpeningFcn, ...
                   'gui_OutputFcn',  @BCseis_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before BCseis is made visible.
function BCseis_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BCseis (see VARARGIN)

% Choose default command line output for BCseis
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BCseis wait for user response (see UIRESUME)
% uiwait(handles.figure1);
% Initialize flags
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global help_path
DATA_READ_FLAG=0;
CWT_COMPUTE_FLAG=0;
POLY_PICK_FLAG=0;
%*****************************************************************
% Set the path for the the help pages
help_path='/Users/cal/matlab/denoising/bc_v1.1/help/';

% --- Outputs from this function are returned to the command line.
function varargout = BCseis_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in tlim_pushbutton.
function tlim_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to tlim_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

if DATA_READ_FLAG == 1
    %  Change the time limits on the seismogram plot
    tstart=str2num(get(handles.tmin_edit,'String'))
    tfinal=str2num(get(handles.tmax_edit,'String'))

    % replot seismogram with new time limits
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');
else
end

if CWT_COMPUTE_FLAG ==1
    %  Change the time limits on the TFR plot
    tstart=str2num(get(handles.tmin_edit,'String'));
    tfinal=str2num(get(handles.tmax_edit,'String'));
    slider_value=get(handles.saturation_slider,'Value');

    delt=1.0./sacdata_new(1).samprate;
    npts=sacdata_new(1).nsamps;
    t=linspace(0,delt.*(npts-1),npts);
    
        % plot the CWT in the TFR_axes1
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
else
end

function tmin_edit_Callback(hObject, eventdata, handles)
% hObject    handle to tmin_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tmin_edit as text
%        str2double(get(hObject,'String')) returns contents of tmin_edit as a double


% --- Executes during object creation, after setting all properties.
function tmin_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tmin_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tmax_edit_Callback(hObject, eventdata, handles)
% hObject    handle to tmax_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tmax_edit as text
%        str2double(get(hObject,'String')) returns contents of tmax_edit as a double


% --- Executes during object creation, after setting all properties.
function tmax_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tmax_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function wavelet_type_edit_Callback(hObject, eventdata, handles)
% hObject    handle to wavelet_type_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wavelet_type_edit as text
%        str2double(get(hObject,'String')) returns contents of wavelet_type_edit as a double


% --- Executes during object creation, after setting all properties.
function wavelet_type_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wavelet_type_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function voices_edit_Callback(hObject, eventdata, handles)
% hObject    handle to voices_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of voices_edit as text
%        str2double(get(hObject,'String')) returns contents of voices_edit as a double


% --- Executes during object creation, after setting all properties.
function voices_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to voices_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pick_with_mouse_pushbutton.
function pick_with_mouse_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to pick_with_mouse_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global wavelet_type nvoices
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

% get the value of the slider position
slider_value=get(handles.saturation_slider,'Value').*0.999;

h2=handles.TFR_axes1;
% Using the mouse, determine a polygon on the t-s plane
[xv,yv]=getline(h2,'closed');

% plot the CWT in TFR_axes1
clear title xlabel ylabel

cla(h2);
set(h2,'NextPlot','add');

% hold on
% imagesc(h2,t, log10(as_new), abs(Wx_new));
% xlim(h2,[tstart tfinal]);
% ylim(h2,[min(log10(as_new)) max(log10(as_new))]);
% h2.YDir='reverse';
% Clim=clim_orig.*(1.0 - slider_value);
% set(h2,'Clim',Clim);
% title(h2,{'CWT Scalogram'},'Rotation',0,'FontSize',14);
% xlabel(h2,{'Time (s)'},'FontSize',12)
% ylabel(h2,'log10 Scale (log10 s)','FontSize',12)
% h2.TitleFontSizeMultiplier = 1.8;
% h2.LabelFontSizeMultiplier=1.8;
% h2.FontWeight='bold';

nverts=length(xv);
fprintf('Polygon vertices\n');
for k=1:nverts;
    fprintf(' x= %g y= %g\n',xv(k),yv(k));
end

POLY_PICK_FLAG=1;

tfrplot(h2,slider_value);
set(h2,'NextPlot','replace');


% --- Executes on button press in exclude_block_pushbutton.
function exclude_block_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to exclude_block_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

%  Apply Block windowing using picked polygon values

if CWT_COMPUTE_FLAG == 1;
    
    % setup to find the polygon mask
    Wx_old=Wx_new;
    as_old=as_new;
    
    % WT is a na x n matrix with ones for the location of the polygon
    WT=poly_mask_zeros(t,as_old,xv,yv);
    
    Wx_new=Wx_old.*WT;
    
    slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update old sacdata
    sacdata_old=sacdata_new;
    
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');

else
end


% --- Executes on button press in apply_pushbutton.
function apply_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to apply_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

%  Apply "Bandpass Blocking" using parameters input into the GUI

if CWT_COMPUTE_FLAG == 1;
    
    %  Gather parameters
    scale_min=str2num(get(handles.scale_min_edit,'String'))
    scale_max=str2num(get(handles.scale_max_edit,'String'))
    thresh=str2num(get(handles.threshold_edit,'String')).*0.01
    
    %  save old CWT
    Wx_old=Wx_new;
    as_old=as_new;
    
%     %  plot scale vs number
%     figure;
%     plot(log10(as_old));
    length(as_old)
    %na
    %  Threshold Wx_old to get Wx_new
    a(1,1:na)=1.0;
    a=a.*(as_old <= scale_min | as_old >= scale_max);
    for k=1:na
        if a(k) == 0
            a(k)=thresh;
        end
    end
    Wx_new=Wx_old.*a';
    
    slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % save old version of sac data
    sacdata_old=sacdata_new;
    
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');

else
end


function scale_min_edit_Callback(hObject, eventdata, handles)
% hObject    handle to scale_min_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of scale_min_edit as text
%        str2double(get(hObject,'String')) returns contents of scale_min_edit as a double


% --- Executes during object creation, after setting all properties.
function scale_min_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scale_min_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function scale_max_edit_Callback(hObject, eventdata, handles)
% hObject    handle to scale_max_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of scale_max_edit as text
%        str2double(get(hObject,'String')) returns contents of scale_max_edit as a double


% --- Executes during object creation, after setting all properties.
function scale_max_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scale_max_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --------------------------------------------------------------------
function file_menu_Callback(hObject, eventdata, handles)
% hObject    handle to file_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function plot_menu_Callback(hObject, eventdata, handles)
% hObject    handle to plot_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function parameters_menu_Callback(hObject, eventdata, handles)
% hObject    handle to parameters_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function calc_menu_Callback(hObject, eventdata, handles)
% hObject    handle to calc_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function refresh_menu_Callback(hObject, eventdata, handles)
% hObject    handle to refresh_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig

%  Refresh the plot to the original seismogram and, if computed, CWT

if DATA_READ_FLAG == 1;
    
    %  load working structures
    sacdata_new=sacdata;
    sacdata_old=sacdata;

    %  update sacnamefile
    sacnamefile=sacdata(1).filename;

    %  Now plot the seismogram in the seismogram window
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    dt=1./sacdata_new(1).samprate;
    np=sacdata_new(1).nsamps;
    tstart=0.0;
    t=linspace(tstart,tstart + dt.*(np-1),np);
    tfinal=t(np);
    
    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');

    % send tstart and tfinal to the TLIM edit boxes
    set(handles.tmin_edit,'String',num2str(tstart));
    set(handles.tmax_edit,'String',num2str(tfinal));
    else;
end

set(handles.saturation_slider,'Value',0);
    
if CWT_COMPUTE_FLAG == 1
    
    % load the CWT
    Wx_new=Wx;
    as_new=as;
    
    slider_value=get(handles.saturation_slider,'Value');
    % plot the CWT in TFR_axes1
    clear title xlabel ylabel clim
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    POLY_PICK_FLAG=0;
    
    tfrplot(h2,slider_value);
    set(h2,'NextPlot','replace');
    
else
end

% --------------------------------------------------------------------
function wavelet_type_menu_Callback(hObject, eventdata, handles)
% hObject    handle to wavelet_type_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function voices_menu_Callback(hObject, eventdata, handles)
% hObject    handle to voices_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function plot_tfrseis_menu_Callback(hObject, eventdata, handles)
% hObject    handle to plot_tfrseis_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Plot the existing Seismogram and CWT on a separate figure
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

slider_value=get(handles.saturation_slider,'Value');

figure('Name','Recent TFR and Seismogram');

subplot(2,1,1);
% plot the CWT in the TFR_axes2
if CWT_COMPUTE_FLAG==1
    hold on
    imagesc(t, log10(as_new), abs(Wx_new));
    xlim([tstart tfinal]);
    ylim([min(log10(as_new)) max(log10(as_new))]);
    ax=gca;
    ax.YDir='reverse';
    Clim=clim_orig.*(1.0 - slider_value).*0.999;
    set(ax,'Clim',Clim);
%     title({'CWT Scalogram'},'Rotation',0,'FontSize',14);
    xlabel({'Time (s)'},'FontSize',12)
    ylabel('log10 Scale (s)','FontSize',12)
    ax.TitleFontSizeMultiplier = 1.8;
    ax.LabelFontSizeMultiplier=1.8;
    ax.FontWeight='bold';
    
    if POLY_PICK_FLAG==1; plot(xv,yv,'-w');end
    
    hold off
end

subplot(2,1,2);
aplot=sacdata_new(1).data(:,1);
np=sacdata_new(1).nsamps;
plot(t,aplot(1:np),'-k');
ax=gca;
axis([tstart tfinal -inf inf]);
xlabel('Time (s)','FontSize',12);
ylabel('log10 Scale (s)','FontSize',12);
ax.TitleFontSizeMultiplier = 1.8;
ax.LabelFontSizeMultiplier=1.8;
ax.FontWeight='bold';

% --------------------------------------------------------------------
function input_sacfile_menu_Callback(hObject, eventdata, handles)
% hObject    handle to input_sacfile_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%*********************************************************************

%  File menu choice - Open a new SAC file, Will begin a new analysis
%                     session.

%*********************************************************************

global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global tstart tfinal

%        Read in SAC waveform data

DATA_READ_FLAG=1;     %  flag for new session
CWT_COMPUTE_FLAG=0;
POLY_PICK_FLAG=0;

%   Call uigetfile menu
[filename,dirname]=uigetfile('*.*');
if filename == 0; DATA_READ_FLAG=0;return;end

sacfile=strcat(dirname,filename);
    
%   Now read in the SAC data files 
%   Use DOE MatSeis readsac, 'sacdata' is a structure with all the data.
nfiles=1;
sacdata=readsac2(sacfile,'l' );
%sacdata
    
%  load working structures
sacdata_new=sacdata;
sacdata_old=sacdata;

%  update sacnamefile
sacnamefile=sacdata(1).filename;

%  Now plot the seismogram in the seismogram window
h1=handles.seis_axes1;

cla(h1);
set(h1,'NextPlot','add');

dt=1./sacdata_new(1).samprate;
np=sacdata_new(1).nsamps;
tstart=0.0;
t=linspace(tstart,tstart + dt.*(np-1),np);
tfinal=t(np);

seisplot(h1,sacdata_new);

set(h1,'NextPlot','replace');

% send tstart and tfinal to the TLIM edit boxes
set(handles.tmin_edit,'String',num2str(tstart));
set(handles.tmax_edit,'String',num2str(tfinal));

% Clear the CWT plot box
h2=handles.TFR_axes1;
cla(h2);
set(h2,'NextPlot','add');

set(handles.saturation_slider,'Value',0);



% --------------------------------------------------------------------
function write_sacfile_menu_Callback(hObject, eventdata, handles)
% hObject    handle to write_sacfile_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global sacdata sacdata_new sacdata_old sacnamefile nfiles

% Get the filename and path
[fname,pathname]=uiputfile('*.sac');

sacdata_new(1).filename=strcat(pathname,'/',fname);

writesac_a(1,sacdata_new,'l');


% --------------------------------------------------------------------
function eight_voices_menu_Callback(hObject, eventdata, handles)
% hObject    handle to eight_voices_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose eight voices
nvoices=8;

set(handles.voices_edit,'String',num2str(nvoices));

% --------------------------------------------------------------------
function sixteen_voices_menu_Callback(hObject, eventdata, handles)
% hObject    handle to sixteen_voices_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose 16 voices
nvoices=16;

set(handles.voices_edit,'String',num2str(nvoices));

% --------------------------------------------------------------------
function thirtytwo_voices_menu_Callback(hObject, eventdata, handles)
% hObject    handle to thirtytwo_voices_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose 32 voices
nvoices=32;

set(handles.voices_edit,'String',num2str(nvoices));

% --------------------------------------------------------------------
function sixtyfour_voices_menu_Callback(hObject, eventdata, handles)
% hObject    handle to sixtyfour_voices_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose 64 voices
nvoices=64;

set(handles.voices_edit,'String',num2str(nvoices));

% --------------------------------------------------------------------
function morlet_menu_Callback(hObject, eventdata, handles)
% hObject    handle to morlet_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose the morlet wavelet
wavelet_type="morlet";

set(handles.wavelet_type_edit,'String',wavelet_type);

% --------------------------------------------------------------------
function mhat_menu_Callback(hObject, eventdata, handles)
% hObject    handle to mhat_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose the Mexican hat wavelet
wavelet_type="mhat";

set(handles.wavelet_type_edit,'String',wavelet_type);

% --------------------------------------------------------------------
function shannon_menu_Callback(hObject, eventdata, handles)
% hObject    handle to shannon_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose the Shannon wavelet
wavelet_type="shannon";

set(handles.wavelet_type_edit,'String',wavelet_type);

% --------------------------------------------------------------------
function hermitian_hat_menu_Callback(hObject, eventdata, handles)
% hObject    handle to hermitian_hat_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices

%  choose the hermitian hat wavelet
wavelet_type="hhat";

set(handles.wavelet_type_edit,'String',wavelet_type);

% --------------------------------------------------------------------
function help_menu_Callback(hObject, eventdata, handles)
% hObject    handle to help_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function help_pages_menu_Callback(hObject, eventdata, handles)
% hObject    handle to help_pages_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%  call the matlab web browser for information on this page
global help_path

overv_page=strcat('file://',help_path,'help.html');

web(overv_page);

% --------------------------------------------------------------------
function compute_cwt_menu_Callback(hObject, eventdata, handles)
% hObject    handle to compute_cwt_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Compute the Continuous Wavelet Transform (CWT) of the Input Signal, if it
% exists

global wavelet_type nvoices
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig

if DATA_READ_FLAG == 1
   
    % setup computation for wavelet transform - first choose data
    delt=1.0./sacdata_new(1).samprate;
    npts=sacdata_new(1).nsamps;
    x=sacdata_new(1).data(:,1);
    
    wavelet_type=get(handles.wavelet_type_edit,'String')
    nvoices=str2num(get(handles.voices_edit,'String'))

    % Compute CWT and show a waitbar
    h3 = waitbar(0.1,'Wavelet Transforming...');
    [Wx_new,as_new] = cwt_fw(x,wavelet_type,nvoices,delt);
    waitbar(0.8,h3,'Wavelet Transforming...'); close(h3)
    [na,n] = size(Wx_new);
    
    if CWT_COMPUTE_FLAG == 0
        Wx=Wx_new;
        as=as_new;
    end
    
    % plot the CWT in TFR_axes1
    clear title xlabel ylabel Clim
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    CWT_COMPUTE_FLAG=1;
    slider_value=0.;
    clim_orig=[0 max(max(abs(Wx_new)))];
    tfrplot(h2,slider_value);
    
    set(h2,'NextPlot','replace');
else
end


% --- Executes on slider movement.
function saturation_slider_Callback(hObject, eventdata, handles)
% hObject    handle to saturation_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global xv yv

% get the value of the slider position
slider_value=get(handles.saturation_slider,'Value').*0.999;

if CWT_COMPUTE_FLAG == 1;
    
    tstart=str2num(get(handles.tmin_edit,'String'));
    tfinal=str2num(get(handles.tmax_edit,'String'));
    
    % plot the CWT in TFR_axes1
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
else
end
    
% --- Executes during object creation, after setting all properties.
function saturation_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to saturation_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in include_block_pushbutton.
function include_block_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to include_block_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global wavelet_type nvoices
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

%  Apply Block windowing using picked polygon values

if CWT_COMPUTE_FLAG == 1
    
    % setup to find the polygon mask
    Wx_old=Wx_new;
    as_old=as_new;
    
    % WT is a na x n matrix with ones for the location of the polygon
    WT=poly_mask_ones(t,as_old,xv,yv);
    
    Wx_new=Wx_old.*WT;
    
    slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update old sacdata before block thresholding
    sacdata_old=sacdata_new;
    
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');

else
end


function threshold_edit_Callback(hObject, eventdata, handles)
% hObject    handle to threshold_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of threshold_edit as text
%        str2double(get(hObject,'String')) returns contents of threshold_edit as a double


% --- Executes during object creation, after setting all properties.
function threshold_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to threshold_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function begin_time_edit_Callback(hObject, eventdata, handles)
% hObject    handle to begin_time_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of begin_time_edit as text
%        str2double(get(hObject,'String')) returns contents of begin_time_edit as a double


% --- Executes during object creation, after setting all properties.
function begin_time_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to begin_time_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function end_time_edit_Callback(hObject, eventdata, handles)
% hObject    handle to end_time_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of end_time_edit as text
%        str2double(get(hObject,'String')) returns contents of end_time_edit as a double


% --- Executes during object creation, after setting all properties.
function end_time_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to end_time_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function estimate_noise_menu_Callback(hObject, eventdata, handles)
% hObject    handle to estimate_noise_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Estimate the noise and standard deviation over a picked time window in
% the seismogram.  The mean and standard deviation of Wavelet coefficient 
% amplitude over the time window will be computed from the CWT.

% In addition, the noise reduction sigma factor will be computed if
% Donoho's Threshold Criterion radiobutton has been pushed.  The "Sigma
% Factor" edit box will then be updated.

global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global M S P
% Get the time window.  If times have not been entered, return an error
% message to the command window.

begtime=str2num(get(handles.begin_time_edit,'String'));
endtime=str2num(get(handles.end_time_edit,'String'));

if (strcmp(begtime,'') || strcmp(endtime,''))
    fprintf('Please Input the begin and ending times for the noise time window\n');
    return
end

delta=1./sacdata_new(1).samprate;
nbeg=round(begtime./delta) + 1;
nend=round(endtime./delta) + 1;
n_noise=nend-nbeg+1;
nlbound=0;

% default sig_fact
sig_fact=str2num(get(handles.sigma_factor_edit,'String'));

% find out if the "ECDF Method" button has been pushed
necdf_flag=get(handles.ecdf_radiobutton,'Value');
% Get the value of noise floor % confidence level
if necdf_flag == 1
    nlbound=str2num(get(handles.noise_lower_bound_edit,'String'));
end

% find out if the "Donoho's Threshold Criterion" radiobutton has been
% pushed
ndono_thresh=get(handles.nlogn_radiobutton,'Value');
if ndono_thresh == 1
    sig_fact=sqrt(2.*log10(n_noise));
    
    % update sigma factor editbox
    set(handles.sigma_factor_edit,'String',num2str(sig_fact));
end

% Calculate the threshold for each scale noise estimate

thresh_calc(nbeg,nend,n_noise,sig_fact,necdf_flag,nlbound);



% --------------------------------------------------------------------
function noise_threshold_menu_Callback(hObject, eventdata, handles)
% hObject    handle to noise_threshold_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Apply the hard threshold method on removing noise from a noise
% estimate made from the data

global wavelet_type nvoices
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global M S P

sigma_factor=str2num(get(handles.sigma_factor_edit,'String'));
M_max=max(abs(M));

% update old CWT TFR
Wx_old=Wx_new;
as_old=as_new;

% hard threshold
W_test=Wx_old;
Wx_new=Wx_old.*(P' < abs(W_test));

slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update sacdata_old
    sacdata_old=sacdata_new;
        
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');



function noise_lower_bound_edit_Callback(hObject, eventdata, handles)
% hObject    handle to noise_lower_bound_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of noise_lower_bound_edit as text
%        str2double(get(hObject,'String')) returns contents of noise_lower_bound_edit as a double


% --- Executes during object creation, after setting all properties.
function noise_lower_bound_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to noise_lower_bound_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sigma_factor_edit_Callback(hObject, eventdata, handles)
% hObject    handle to sigma_factor_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sigma_factor_edit as text
%        str2double(get(hObject,'String')) returns contents of sigma_factor_edit as a double


% --- Executes during object creation, after setting all properties.
function sigma_factor_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sigma_factor_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function snr_detector_menu_Callback(hObject, eventdata, handles)
% hObject    handle to snr_detector_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Apply an empirical event detector by dividing the CWT field by the noise
% estimate.  A free paramenter is the noise lower bound.

global wavelet_type nvoices
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global M S P

% update TFR
Wx_old=Wx_new;
as_old=as_new;

nlbound=str2num(get(handles.noise_lower_bound_edit,'String')).*0.01;
M_max=max(abs(M));
Wx_new=Wx_old./(M'+nlbound.*M_max);

slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update sacdata_old
    sacdata_old=sacdata_new;
    
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');


% --------------------------------------------------------------------
function soft_threshold_menu_Callback(hObject, eventdata, handles)
% hObject    handle to soft_threshold_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Apply the soft threshold method on removing noise from a noise
% estimate made from the data

global wavelet_type nvoices
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global M S P

sigma_factor=str2num(get(handles.sigma_factor_edit,'String'));
M_max=max(abs(M));

% update old TFR
Wx_old=Wx_new;
as_old=as_new;

% soft threshold
W_test=abs(Wx_old);
Wx_new=sign(Wx_old).*(W_test - P').*(P' < W_test);

slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update sacdata_old
    sacdata_old=sacdata_new;
    
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');


% --- Executes on button press in nlogn_radiobutton.
function nlogn_radiobutton_Callback(hObject, eventdata, handles)
% hObject    handle to nlogn_radiobutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of nlogn_radiobutton



function log10scale_edit_Callback(hObject, eventdata, handles)
% hObject    handle to log10scale_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of log10scale_edit as text
%        str2double(get(hObject,'String')) returns contents of log10scale_edit as a double

% scale calculator panel - get the number in the log10 edit box
scale_log10=str2num(get(handles.log10scale_edit,'String'));

% compute linear value
scale_lin=10.^scale_log10;
set(handles.linear_edit,'String',num2str(scale_lin));



% --- Executes during object creation, after setting all properties.
function log10scale_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to log10scale_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function linear_edit_Callback(hObject, eventdata, handles)
% hObject    handle to linear_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of linear_edit as text
%        str2double(get(hObject,'String')) returns contents of linear_edit as a double

% scale calculator panel - get the number in the linear edit box
scale_lin=str2num(get(handles.linear_edit,'String'));

% compute log10
if scale_lin > 0;
    scale_log10=log10(scale_lin);
    set(handles.log10scale_edit,'String',num2str(scale_log10));
else
    set(handles.log10scale_edit,'String','*****');
end
    



% --- Executes during object creation, after setting all properties.
function linear_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to linear_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Undo_menu_Callback(hObject, eventdata, handles)
% hObject    handle to Undo_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig

%  Undo the last operation and plot seismogram and TFR

if DATA_READ_FLAG == 1
    
    %  load working structures
    sacdata_new=sacdata_old;

    %  update sacnamefile
    sacnamefile=sacdata(1).filename;

    %  Now plot the seismogram in the seismogram window
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    dt=1./sacdata_new(1).samprate;
    np=sacdata_new(1).nsamps;

%     tstart=0.0;
%     t=linspace(tstart,tstart + dt.*(np-1),np);
%     tfinal=t(np);
    
    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');

%     % send tstart and tfinal to the TLIM edit boxes
%     set(handles.tmin_edit,'String',num2str(tstart));
%     set(handles.tmax_edit,'String',num2str(tfinal));
    else;
end

set(handles.saturation_slider,'Value',0);
    
if CWT_COMPUTE_FLAG == 1
    
    % load the CWT
    Wx_new=Wx_old;
    as_new=as_old;
    
    slider_value=get(handles.saturation_slider,'Value');
    % plot the CWT in TFR_axes1
    clear title xlabel ylabel clim
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    POLY_PICK_FLAG=0;
    
    tfrplot(h2,slider_value);
    set(h2,'NextPlot','replace');
    
else
end


% --------------------------------------------------------------------
function hard_threshold_signal_menu_Callback(hObject, eventdata, handles)
% hObject    handle to hard_threshold_signal_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Apply the hard threshold method on removing signal from a noise
% estimate made from the data

% M is the noise estimate

global wavelet_type nvoices
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global M S P

sigma_factor=str2num(get(handles.sigma_factor_edit,'String'));
M_max=max(abs(M));

% update old CWT TFR
Wx_old=Wx_new;
as_old=as_new;

% hard threshold
W_test=Wx_old;
Wx_new=Wx_old.*(P' > abs(W_test));

slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update sacdata_old
    sacdata_old=sacdata_new;
        
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');



% --------------------------------------------------------------------
function soft_threshold_signal_menu_Callback(hObject, eventdata, handles)
% hObject    handle to soft_threshold_signal_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Apply the soft threshold method on removing signal from a noise
% estimate made from the data

% M is the noise estimate

global wavelet_type nvoices
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global sacdata sacdata_new sacdata_old sacnamefile nfiles
global M S P

sigma_factor=str2num(get(handles.sigma_factor_edit,'String'));
M_max=max(abs(M));

% update old TFR
Wx_old=Wx_new;
as_old=as_new;

% soft threshold
W_test=Wx_old;
Wx_new=sign(Wx_old).* P'.*(P' <= abs(W_test)) + W_test.*(P' > abs(W_test));

slider_value=get(handles.saturation_slider,'Value');
    % Plot the thresholded CWT
    clear title xlabel ylabel
    h2=handles.TFR_axes1;
    cla(h2);
    set(h2,'NextPlot','add');
    
    tfrplot(h2,slider_value);

    set(h2,'NextPlot','replace');
    
    % Compute the inverse CWT and plot new seismogram
    h3 = waitbar(0.1,'Inverse Wavelet Transforming...');
    anew=cwt_iw(Wx_new,wavelet_type,nvoices);
    waitbar(0.8,h3,'Inverse Wavelet Transforming...'); close(h3)
    
    % update sacdata_old
    sacdata_old=sacdata_new;
    
    npts=sacdata_new(1).nsamps;
    sacdata_new(1).data(1:npts,1)=anew(1:npts);
    
    h1=handles.seis_axes1;

    cla(h1);
    set(h1,'NextPlot','add');

    seisplot(h1,sacdata_new);

    set(h1,'NextPlot','replace');


% --- Executes on button press in ecdf_radiobutton.
function ecdf_radiobutton_Callback(hObject, eventdata, handles)
% hObject    handle to ecdf_radiobutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ecdf_radiobutton
