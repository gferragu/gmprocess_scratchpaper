function [WT] = poly_mask_ones(x,y1,xv,yv)
%poly_mask_ones - compute a mask for finding points within a 2D polygon
PRINT_FLAG=0;
% find the minimum and maximum coordinates of the polygon
xmin=min(xv);
xmax=max(xv);
ymin=min(yv);
ymax=max(yv);

y=log10(y1);
nx=length(x);
ny=length(y);

if PRINT_FLAG==1
fprintf('xmin= %g xmax= %g nx= %i ymin= %g ymax= %g ny= %i\n',xmin,xmax,nx,ymin,ymax,ny);
fprintf('x(1)= %g x(nx)= %g y(1)= %g y(ny)= %g\n',x(1),x(nx),y(1),y(ny));
end

% first make sure that the points are within the arrays
assert(xmin >= x(1) && xmax <= x(nx));
assert(ymin >= y(1) && ymax <= y(ny));

% now find the index of the closest x and y positions in the x and y vectors
[dxmin1,nx_min]=min(abs(x - xmin));
xmin1=xmin+dxmin1;
[dxmax1,nx_max]=min (abs(x - xmax));
xmax1=xmax+dxmax1;
[dymin1,ny_min]=min(abs(y - ymin));
ymin1=ymin+dymin1;
[dymax1,ny_max]=min(abs(y - ymax));
ymax1=ymax+dymax1;

% make sure there is only one point
assert(length(nx_min) == 1);
assert(length(nx_max) == 1);
assert(length(ny_min) == 1);
assert(length(ny_max) == 1);

if PRINT_FLAG==1
fprintf('xmin xmin1 nx_min xmax xmax1 nx_max\n');
fprintf('%g %g %i %g %g %i\n',xmin,xmin1,nx_min,xmax,xmax1,nx_max);

fprintf('ymin ymin1 ny_min ymax ymax1 ny_max\n');
fprintf('%g %g %i %g %g %i\n',ymin,ymin1,ny_min,ymax,ymax1,ny_max);
end

% generate a grid of ordered points (xt,yt) within the rectangular array
% that encloses the polygon
npx=nx_max - nx_min +1;
npy=ny_max - ny_min +1;

% initialize arrays
ntotal=npx.*npy;
xt(1:ntotal)=0.0;
yt(1:ntotal)=0.0;
n_test(1:ntotal,1:2)=0;

WT(1:ny,1:nx)=0.0;

np=0;
for ky=1:npy;
    for kx=1:npx
        np=np+1;
        kx1=nx_min+kx-1;
        ky1=ny_min+ky-1;
        xt(np)=x(kx1);
        yt(np)=y(ky1);
        N_test(np,1)=kx1;
        N_test(np,2)=ky1;
    end
end

if PRINT_FLAG==1
fprintf('ntotal = %i  np= %i\n',ntotal,np);
end

% test to see if the (xt,yt) are in the polygon
IN=inpolygon(xt,yt,xv,yv);

% Create the mask array WT
for k=1:ntotal
    if IN(k)==1
        % this is a point in the polygon
        WT(N_test(k,2),N_test(k,1))=1.0;
    end
end

end

