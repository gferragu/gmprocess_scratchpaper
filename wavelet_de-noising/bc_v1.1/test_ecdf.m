function test_ecdf
% test the ecdf function using a Gaussian distribution

% calculate a Gaussian distribution of data

mu=250.;
sigma=25.0;
pd=makedist('Normal',mu,sigma);
x=linspace(0,500,501);
y=pdf(pd,x);

figure('Name','Normal Distribution');
plot(x,y,'-k');

y=random(pd,1,500);
[f,x1]=ecdf(y);

figure('Name','Empirical CDF');
plot(x1,f,'-k');

end

