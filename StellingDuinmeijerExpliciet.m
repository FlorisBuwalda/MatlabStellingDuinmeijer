function StellingDuinmeijerExpliciet
%Implementation of Stelling & Duinmeijer explicit staggered momentum
%conservative scheme without slope limiters, with the bed friction term taken implicitly.
%Plot title shows t = time step,
%tv = measure of total momentum in the system.
%Surface plot displays water height colored by momentum

%% parameters
n = 60;                  % grid size (both dimensions)
g = 9.8;                 % gravitational constant
tstep = .1;             % maximum timestep   
dt=.00001;                % first step is maximum timestep 
dx = 1.0;                % inter grid distance in x- direction   
dy = 1.0;                % inter grid distance in y- direction 
cf=0;                    % Bottom friction factor
nplotstep = 1;           % number of timesteps per plot step
ndrops = 3;              % maximum number of water drops
dropstep = 10;            % drop interval
Hstart = 2;
D = droplet(1.5,21); % simulate a water drop
rand=1;                  % Uncomment for consistent water drop location and size

%% Initialize graphics
[surfplot,top,restart,quit] = initgraphics(n+2,Hstart);

%% Outer loop, restarts.
while get(quit,'value') == 0
     set(restart,'value',0) 
   H = Hstart*ones(n+3,n+3);   U = zeros(n+3,n+3);  V = zeros(n+3,n+3); Hx = U; Hy = V;
   
   %set boundary values
       H(1,:)=0; Hx(1,:)=0;  U(1,:)=0; %i=1 is dummy row
       H(:,1)=0; Hy(:,1)=0;  V(:,1)=0; %j=1 is also dummy column
    % H(n/2+.5:n/2+20.5,n/2+5.5:n/2+20.5)=2.7;
     
   ndrop = ceil(rand*ndrops);
   nstep = 0;
   %% Inner loop, time integration
 
   while get(restart,'value')==0 && get(quit,'value')==0
       nstep = nstep + 1;
   %  Random water drops
        if mod(nstep,dropstep) == 0 && nstep <= ndrop*dropstep
            w = size(D,1);           
            i = ceil(rand*(n-w))+(1:w);
            j = ceil(rand*(n-w))+(1:w);
            H(i,j) = H(i,j) + (1+4*rand)/5*D;
        end

        %Velocity = 0 at the boundary
        U(:,1)=0; U(:,end)=0; V(1,:)=0; V(end,:)=0;
        %U(2,2)=0; V(2,2)=0;
        % Uncomment to move top left boundary  
        % U(1:2*n/4,1:2*n/4)=0;
        % V(1:2*n/4,1:2*n/4)=0;

%% Define Hx and Hy by dynamic upwinding depending on adjacent velocity
        i=2:n+2;
        j=2:n+3;
        
        Hx(j,i) = (U(j,i)>0).*H(j,i) + (U(j,i)<0).*H(j,i+1) + (U(j,i)==0).*max(H(j,i),H(j,i+1)); 
        
        i=2:n+3;
        j=2:n+2;
         
        Hy(j,i) = (V(j,i)>0).*H(j,i) + (V(j,i)<0).*H(j+1,i) + (V(j,i)==0).*max(H(j,i),H(j+1,i)); 
        
        
     %Boundaries equal to neighboring value
      Hx(2:end,1) = H(2:end,2);
      Hy(1,2:end) = H(2,2:end);
      Hx(2:end,end) = H(2:end,end-1);
      Hy(end,2:end)= H(end-1,2:end);
      
      
      %% Update U
       i =2:n+2;
       j= 2:n+3;
             
       U(j,i) = (U(j,i)     - g*dt/dx*(H(j,i+1)-H(j,i)) ...                                                                           %Gravity term
                            - dt/dx* (U(j,i)-U(j,i-1))      .*  (Hx(j,i).*U(j,i)  +Hx(j,i-1)  .*U(j,i-1))     ./(H(j,i)+H(j,i+1))...  %advection U    %- dt/dx* (U(j,i)-U(j,i-1)).*(U(j,i)+U(j,i-1))/2 ...            % 
                            - dt/dy* (U(j,i)-U(j-1,i))     .*(Hy(j-1,i).*V(j-1,i)+Hy(j-1,i+1).*V(j-1,i+1))   ./(H(j,i)+H(j,i+1)))... %advection V    %- dt/dy* (U(j,i)-U(j-1,i)).*(V(j-1,i)+V(j-1,i-1))/2) ...       %
                            ./(1+dt*cf*sqrt((U(j,i).^2+V(j,i).^2))./(H(j,i)+H(j,i+1)));                                               %Bottom friction (implicit)
                        
       %% Update V 
       i=2:n+3;
       j=2:n+2;
       
       V(j,i) =  (V(j,i) - g*dt/dy*(H(j+1,i)-H(j,i))   ...%                                                                            %Gravity term
                               - dt/dx* (V(j,i)-V(j,i-1)) .*  (Hx(j,i-1) .*U(j,i-1)+Hx(j+1,i-1).*U(j+1,i-1))./  (H(j,i)+H(j+1,i))...  %advection U -dt/dx* (V(j,i)-V(j,i-1)).*(U(j,i-1)+U(j-1,i-1))/2 ...            %
                               - dt/dy* (V(j,i)-V(j-1,i))  .* (Hy(j,i)   .*V(j,i)  +Hy(j-1,i)  .*V(j-1,i))  ./  (H(j,i)+H(j+1,i)))...%advection V- dt/dy* (V(j,i)-V(j-1,i)).*(V(j,i)+V(j-1,i))/2) ...           %
                               ./(1+dt*cf*sqrt((U(j,i).^2+V(j,i).^2))./(H(j,i)+H(j+1,i)));                                            %Bottom friction (implicit) 
   
        %% Update H
    i=2:n+3;
    j=2:n+3;

    H(j,i) = H(j,i) - dt/dx*(Hx(j,i).*U(j,i)-Hx(j,i-1).*U(j,i-1))...
                    - dt/dy*(Hy(j,i).*V(j,i)-Hy(j-1,i).*V(j-1,i));
              
        %% Update plot
       if mod(nstep,nplotstep) == 0
          C = abs(U(i,j)) + abs(V(i,j));  % Color shows momemtum
          t = nstep;
          tv = norm(C,'fro');
          set(surfplot,'zdata',H(i,j),'cdata',C);
          set(top,'string',sprintf('t = %6.2f,  tv = %6.2f',t,tv))
          drawnow
         
       end
       
       %% CFL Condition
      % Advection free CFL condition
      %dt = min(tstep,.8*dx/(2*sqrt(2)*sqrt(g*max(max(H))))); %safety factor of .8
     %dt = .01;
      % CFL condition with velocity
      dt = min(tstep,    ...
         .2*dx/max(max(  ...    %safety factor of .3
           max( 10*abs(U(j,i)+U(j-1,i)./(2*H(j,i)))    + sqrt(g*H(j,i)) ,         ...
                10*abs(V(j,i)+V(j,i-1)./(2*H(j,i)))   + sqrt(g*H(j,i)) ...
           ))));
       
      % if nstep>900
        %   breakpoint=1;
           U(1:6,1:6)
      % end
       
       %Restart if H is NaN or inf
        if max(max(isnan(H)|isinf(H))) 
            break 
        end  % Unstable, restart

   end
end
close(gcf)
  
  function D = droplet(height,width)
% DROPLET  2D Gaussian
% D = droplet(height,width)
   [x,y] = ndgrid(-1:(2/(width-1)):1);
   D = height*exp(-5*(x.^2+y.^2));
% ------------------------------------
function [surfplot,top,restart,quit] = initgraphics(n,Hstart);
% INITGRAPHICS  Initialize graphics for waterwave.
% [surfplot,top,restart,quit] = initgraphics(n)
% returns handles to a surface plot, its title, and two uicontrol toggles.
   clf
   shg
   set(gcf,'menubar','none','numbertitle','off','name','Waterwave') 
   x = (0:n-1)/(n-1);
   surfplot = surf(x,x,ones(n,n),zeros(n,n));
   grid off
   axis([0 1 0 1 Hstart-2 Hstart+3])
   caxis([-1 1])
   shading faceted
   c = (1:64)'/64;
   %cyan = [0*c c c];
   %colormap(default)
   colormap default
   top = title('xxx');
   restart = uicontrol('position',[20 20 80 20],'style','toggle','string','restart');
   quit = uicontrol('position',[120 20 80 20],'style','toggle','string','close');
