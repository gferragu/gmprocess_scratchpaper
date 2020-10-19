function seisplot(h1,s)
%  Plot the seismogram in the sac data structure s
global tstart tfinal
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG

% tstart
% tfinal

aplot=s(1).data(:,1);
dt=1./s(1).samprate;
np=s(1).nsamps;
tstart1=0.0;

t=linspace(tstart1,tstart1 + dt.*(np-1),np);
% hold on
plot(h1,t,aplot(1:np),'-k');
% xlabel(h1,'Time (s)');
% hold off

axis(h1,[tstart tfinal -inf inf]);
xlabel(h1,{'Time (s)'},'FontSize',12)
ylabel(h1,'log10 Scale (s)','FontSize',12)
h1.TitleFontSizeMultiplier = 1.8;
h1.LabelFontSizeMultiplier=1.8;
h1.FontWeight='bold';

end

