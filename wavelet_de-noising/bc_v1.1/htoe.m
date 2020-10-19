%HTOE	Human time to epochal time.
%	EPOCHAL = HTOE ( TIME ) returns the epochal time of the given human 
%	time. Human time is a 6 element row vector: 
%	  [ year month day hour minute second ] 
%	TIME may be a matrix of human times with 1 time value per row. In this 
%	case a column vector of epochal time is returned. 
%
%	See also ETOH, YMD.

%	MatSeis 1.6
%	header and specs from MatSeis, Code by C.A. Langston
%   7/06/04

function epochal=htoe( time )
%time
[nrows,ncols]=size(time);
year(1:nrows)=time(1:nrows,1);
month(1:nrows)=time(1:nrows,2);
day(1:nrows)=time(1:nrows,3);
hour(1:nrows)=time(1:nrows,4);
minute(1:nrows)=time(1:nrows,5);
second(1:nrows)=time(1:nrows,6);

epochal(1:nrows)=0.;

mday=[31 28 31 30 31 30 31 31 30 31 30 31];

for k=1:length(day);
    
    %  First find the Julian day within the year
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
    
    %  test for February 29 on a non leap year
    if (month(k) == 2 && day(k) == 29) && leap == 0;
        fprintf('****Error from htoe.m -  February 29th input for a non leap year!!****\n');
        return;
    else;
    end;
    
    jday=0;
    if month(k) >= 2;
        for j=1:month(k)-1;
            jday=jday + mday(j);
        end;
        jday=jday+day(k);
    else;
        jday=jday+day(k);
    end;
    if month(k) == 2 && day(k) == 29; jday=jday+leap;end;
    if month(k) >= 3; jday=jday+leap; end;
    
    %  found Julian day within the year
    %  Now find Julian days from [1970 1 1 0 0 0 ]
    
    dyear=year(k)-1970;
    
    if dyear >= 1;
        for j=1:dyear;
            leap=0;
            %  test for leap year
            test_year=1970 + j -1;
            dyear1=fix(test_year/4);
            r_dyear1=test_year - dyear1*4;
        
            dyear2=fix(test_year/100);
            r_dyear2=test_year - dyear2*100;

            dyear3=fix(test_year/400);
            r_dyear3=test_year - dyear3*400;

            if r_dyear1 == 0; leap=1;end;
            if r_dyear2 == 0; leap=0;end;
            if r_dyear3 == 0; leap=1;end;
            
            if leap == 0; mdays=365; else; mdays=366; end;
            
            jday=jday + mdays;
        end;
    else;
    end;
    
    julday(k)=jday;
    
end;
    
%  calculate epochal time in seconds
    
epochal=(julday-1)*86400 + hour*3600 + minute*60 + second;


            