function writesac(n,s,endian)
%  writesac(n,s,endian)
%      Write all SAC format files to disk
%          s = SAC file structure ala readsac2
%          endian = 'b' for bigendian format; = 'l' for little endian
%          format
%
%      Only evenly sampled time series are supported in this routine
%      C.A. Langston - 7/04/05

for m=1:n

sacfile=s(m).filename;

%  open sacfile and write the data
fid=fopen(sacfile,'w');

a_count=fwrite(fid,s(m).headerA,'float32');
%s(m).headerA

b_count=fwrite(fid,s(m).headerB,'int32');
%s(m).headerB

c_count=fwrite(fid,s(m).headerC,'char');
%s(m).headerC(1,1:192)

d_count=fwrite(fid,s(m).data(:,1),'float32');
%s(m).data(1,:)

fclose(fid);

end