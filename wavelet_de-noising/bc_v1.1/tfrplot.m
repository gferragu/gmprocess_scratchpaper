function tfrplot(h2,slider_value)
%tfrplot - plot the most recent tfr image
global DATA_READ_FLAG CWT_COMPUTE_FLAG POLY_PICK_FLAG
global tstart tfinal
global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
global xv yv

hold on
imagesc(h2,t, log10(as_new), abs(Wx_new));
xlim(h2,[tstart tfinal]);
ylim(h2,[min(log10(as_new)) max(log10(as_new))]);
h2.YDir='reverse';
Clim=clim_orig.*(1.0 - slider_value).*0.999;
set(h2,'Clim',Clim);
% title(h2,{'CWT Scalogram'},'Rotation',0,'FontSize',14);
xlabel(h2,{'Time (s)'},'FontSize',12);
ylabel(h2,'log10 Scale (s)','FontSize',12);
h2.TitleFontSizeMultiplier = 1.8;
h2.LabelFontSizeMultiplier=1.8;
h2.FontWeight='bold';

if POLY_PICK_FLAG==1; plot(h2,xv,yv,'-w');end
hold off

end

