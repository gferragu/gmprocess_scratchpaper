%YMD	Yeardoy to year, month, and day.
%	[YEAR,MONTH,DAY] = YMD ( YEARDOY ) returns the YEAR, MONTH, and DAY of 
%	the given YEARDOY. YEARDOY is the year and day-of-year in the format 
%	yyyyddd. 
%	YEARDOY may be a column vector of yeardoys. In this case YEAR, MONTH, 
%	and DAY are column vectors of the same size.
%
%	See also ETOH, HTOE, YEARDOY.

%	Uses the header of MatSeis-1.6 but code written by
%   C.A. Langston 7/5/04

function [year,month,day]=ymd(yeardoy)

%  calculate year and julian day
year=fix(yeardoy/1000);
jday=yeardoy - year*1000;

for k=1:length(yeardoy);
    
    leap=0;
    %  test for leap year
    dyear1=fix(year(k)/4);
    r_dyear1=year(k)-dyear1*4;

    dyear2=fix(year(k)/100);
    r_dyear2=year(k)-dyear2*100;

    dyear3=fix(year(k)/400);
    r_dyear3=year(k)-dyear3*400;

    if r_dyear1 == 0; leap=1;end;
    if r_dyear2 == 0; leap=0;end;
    if r_dyear3 == 0; leap=1;end;

    %  generate dates
    mon(1:366)=0;
    dy(1:366)=0;
    if leap == 0;   % no leap year
        mon(1:31)=1;        % Jan
        dy(1:31)=1:31;

        mon(32:59)=2;       % Feb
        dy(32:59)=1:28;
    
        mon(60:90)=3;       % March
        dy(60:90)=1:31;
    
        mon(91:120)=4;      % April
        dy(91:120)=1:30;
    
        mon(121:151)=5;     % May
        dy(121:151)=1:31;
    
        mon(152:181)=6;     % June
        dy(152:181)=1:30;
    
        mon(182:212)=7;     % July
        dy(182:212)=1:31;
    
        mon(213:243)=8;     % Aug
        dy(213:243)=1:31;
        
        mon(244:273)=9;     % Sept
        dy(244:273)=1:30;
    
        mon(274:304)=10;    % Oct
        dy(274:304)=1:31;
    
        mon(305:334)=11;    % Nov
        dy(305:334)=1:30;
    
        mon(335:365)=12;    % Dec
        dy(335:365)=1:31;
    
    else;       % leap year
    
        mon(1:31)=1;        % Jan
        dy(1:31)=1:31;
    
        mon(32:60)=2;       % Feb
        dy(32:60)=1:29;
    
        mon(61:91)=3;       % Mar
        dy(61:91)=1:31;
    
        mon(92:121)=4;      % April
        dy(92:121)=1:30;
    
        mon(122:152)=5;     % May
        dy(122:152)=1:31;
    
        mon(153:182)=6;     % June
        dy(153:182)=1:30;
    
        mon(183:213)=7;     % July
        dy(183:213)=1:31;
    
        mon(214:244)=8;     % Aug
        dy(214:244)=1:31;
    
        mon(245:274)=9;     % Sept
        dy(245:274)=1:30;
    
        mon(275:305)=10;    % Oct
        dy(275:305)=1:31;
    
        mon(306:335)=11;    % Nov
        dy(306:335)=1:30;
    
        mon(336:366)=12;    % Dec
        dy(336:366)=1:31;
    
    end;

    month(k)=mon(jday(k));
    day(k)=dy(jday(k));

end;

year=year';
month=month';
day=day';


