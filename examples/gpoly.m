function g=gpoly(x0,z0,xcorn,zcorn,ncorn,rho)

% g=gpoly(x0,z0,xcorn,zcorn,ncorn,rho)
% computes the vertical attraction of a 2-dimensional body with polygonal
% cross section. Axis are right-handed system with y-axis parallel to long
% direction of body and z-axis vertical, positive downward.
% after Blakely, 
% Potential Theory in Gravity and Magnetic Applications, 1995, C.U.P., p 378
% Observation point is (x0,z0). Arrays xcorn & zcorn (each of length ncorn)
% contain the coordinates of the polygon corners, arranged in clockwise
% order when viewed with x-axis to right. Density of body is rho.
% all distance parameters in units of km, rho in units of kg/m^3
% Output = vertical attraction of gravity g, in mgal

% parameters
gamma=6.673e-11;
si2mg=1e5;
km2m=1e3;

% run
sum=0;
for n=1:ncorn
    if n==ncorn
        n2=1;
    else
        n2=n+1;
    end
    
    x1=xcorn(n)-x0;
    z1=zcorn(n)-z0;
    x2=xcorn(n2)-x0;
    z2=zcorn(n2)-z0;
    r1sq=x1*x1+z1*z1;
    r2sq=x2*x2+z2*z2;
    
    if r1sq==0
        break
        disp('Field point on corner')
    elseif r2sq==0
        break
        disp('Field point on corner')
    end
    
    denom=z2-z1;
    
    if denom==0
        denom=1E-6;
    end
    alpha=(x2-x1)/denom;
    beta=(x1*z2-x2*z1)/denom;
    factor=beta/(1+alpha*alpha);
    term1=0.5*(log(r2sq)-log(r1sq));
    term2=atan2(z2,x2)-atan2(z1,x1);
    sum=sum+factor*(term1-alpha*term2);
end

g=2*rho*gamma*sum*si2mg*km2m;