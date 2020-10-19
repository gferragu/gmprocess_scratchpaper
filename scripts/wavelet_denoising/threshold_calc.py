#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:46:51 2020

@author: gabriel
"""


#%% MATLAB Code to reproduce

# function thresh_calc(nbeg,nend,n_noise,sig_fact,necdf_flag,nlbound)
# % Calculate the threshold using one of two methods
# %   P=mean|W| + c sigma
# %   P is computed from the empirical cdf of the noise signal at some
# %        desired confidence level
# %
# %   Experimental version for bc v1.1,  March 6, 2019
# % 
# global Wx as Wx_old Wx_new as_old as_new t na n clim_orig
# global M S P

# if necdf_flag == 1
    
#     % Compute empirical cdf statistics and noise threshold
#     [nrow,ncol]=size(Wx_new)
#     conf=1.0 - nlbound.*0.01;
# %     For each row in the matrix
#     for k=1:nrow
#         W(1:n_noise)=abs(Wx_new(k,nbeg:nend))';
#         [f,x]=ecdf(W);
        
# %         % plot every 10th cdf
# %         kmod=floor(k/10);
# %         if k == 1 || k == kmod.*10
# %             scale=as_new(k);
# %             str_scale=num2str(scale);
# %             figure;
# %             plot(x,f);
# %             tdum=strcat('ECDF for k=',num2str(k),' scale=',str_scale);
# %             title(tdum);
# %             xlabel('Data Value');
# %             ylabel('Probability');
# %         end
        
#         P(k)=interp1(f,x,conf);
#     end
#         M=mean(abs(Wx_new(:,nbeg:nend)'));
# %         P=P';
        
#     % plot the results in a figure
#     figure('Name','ECDF Threshold');
#     hold on
#     aslg=log10(as_new);
# %     length(M)
# %     length(P)
# %     length(aslg)
#     plot(aslg,M,'-k');
#     plot(aslg,P,'-r');

#     hold off

#     xlabel('log10 Scale (s)');
#     ylabel('Coefficient Amplitude');
#     legend('mean','threshold');
#     tdum=strcat(num2str(conf.*100),'% Confidence Level');
#     title(tdum);
    
# else
    
#     % Compute Gaussian noise statistics and noise threshold
    
#     M=mean(abs(Wx_new(:,nbeg:nend)'));
#     S=std(abs(Wx_new(:,nbeg:nend)'));
#     P=M + sig_fact.*S;
    
#     Ekur=sqrt(.9).*(kurtosis(abs(Wx_new(:,nbeg:nend)'))-3.0)./sqrt(24.0./n_noise);
    
#     % plot the results in a figure
#     % changed 2/19/19 to show the Threshold defined by sig_fact
#     figure('Name','Noise Mean and Threshold');
#     hold on
#     aslg=log10(as_new);
#     plot(aslg,M,'-k');
#     plot(aslg,P,'-r');

#     hold off

#     xlabel('log10 Scale (s)');
#     ylabel('Coefficient Amplitude');
#     legend('mean','threshold');

#     % plot the Excess kurtosis statistic in a figure
#     figure('Name','Noise Estimate Excess Kurtosis');
#     aslg=log10(as_new);
#     naslg=length(aslg);
#     hold on
#     plot(aslg,Ekur,'-k');
#     plot([aslg(1) aslg(naslg)],[1.0 1.0],'-k');
#     plot([aslg(1) aslg(naslg)],[-1.0 -1.0],'-k');
#     hold off
#     xlabel('log10 Scale (s)');
#     ylabel('Non-Gaussianity');
#     axis([-2.5 2.5 -50 50]);
#     grid on
    
# end
# end

#%% Python Implementation

def threshold_calc(nbeg,nend,n_noise,sig_fact,necdf_flag,nlbound):
    
    
    
    
    return