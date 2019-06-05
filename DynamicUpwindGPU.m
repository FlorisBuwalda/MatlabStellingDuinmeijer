function DynamicUpwindGPU
%Implementation of Stelling & Duinmeijer explicit staggered 
%scheme without slope limiters, with the bed friction term taken implicitly.
%Plot title shows t = time step,
%tv = measure of total momentum in the system.
%dt = current simulation timestep
%tc = calculation time between plot steps
%Surface plot displays water height colored by momentum
%

G = gpuDevice();
%% parameters
n=1000;          % grid size (both dimensions)
L=100;          %Domain length
W=100;          %Domain width
Hstart = 1;    %Rest water depth                  
g = gpuArray(single(9.8));                 % gravitational constant
tstep = gpuArray(single(1));             % maximum timestep   
dt=.00001;                % first step is maximum timestep 
dx = gpuArray(single(W/(n+1)));                % inter grid distance in x- direction   
dy = gpuArray(single(L/(n+1)));                % inter grid distance in y- direction 
cf=gpuArray(single(0));                    % Bottom friction factor
nplotstep = gpuArray(int16(10));           % number of timesteps per plot step
ndrops = gpuArray(int16(1));              % maximum number of water drops
dropstep = gpuArray(int16(5));            % drop interval
D = droplet(1.5*Hstart,floor(floor(n/3)));    % simulate a water drop
%rand=1;                  % Uncomment for consistent water drop location and size
xq = gpuArray(5:ceil(n/100):n);
timer =gpuArray(single(1));
safety=gpuArray(single(.9));
tc=0;
%% Initialize graphics
[surfplot,top,restart,quit] = initgraphics(n+2,Hstart);

%% Outer loop, restarts.
while get(quit,'value') == 0
     set(restart,'value',0) 
   H = Hstart*gpuArray.ones(n+3,n+3,'single');   U = gpuArray.zeros(n+3,n+3,'single');  V = gpuArray.zeros(n+3,n+3,'single'); Hx = U; Hy = V;
   Upos = gpuArray.ones(n+3,n+3,'logical');  Vpos=Upos; Uneg=gpuArray.zeros(n+3,n+3,'logical'); Vneg=Uneg; 
   %H = gpuArray(H); U=gpuArray(U); V=gpuArray(V);Hx = gpuArray(Hx); Hy = gpuArray(Hy);
   %set boundary values
       H(1,:)=0; Hx(1,:)=0;  U(1,:)=0; %i=1 is dummy row
       H(:,1)=0; Hy(:,1)=0;  V(:,1)=0; %j=1 is also dummy column
    % H(n/2+.5:n/2+20.5,n/2+5.5:n/2+20.5)=2.7;
     
   %ndrop = ceil(rand*ndrops);
   nstep = gpuArray(int16(0));
   %% Inner loop, time integration
 
   while get(restart,'value')==0 && get(quit,'value')==0
       nstep = nstep + 1;
     
   %  Random water drops
        if mod(nstep,dropstep) == 0 && nstep <= ndrops*dropstep
            w = size(D,1);           
            i = ceil(rand*(n-w))+(1:w);
            j = ceil(rand*(n-w))+(1:w);
            H(i,j) = H(i,j) + (1+4*rand)/5*D;
        end
        
        %Velocity = 0 at the boundary
        U(:,1)=0; U(:,end)=0; V(1,:)=0; V(end,:)=0;
        %U(2,2)=0; V(2,2)=0;
        
        
%% Define Hx and Hy by dynamic upwinding depending on adjacent velocity
        i=2:n+2;
        j=2:n+3;
        
        Hx(j,i) = (Upos(j,i)).*H(j,i) + (Uneg(j,i)).*H(j,i+1) ;%+ (U(j,i)==0).*max(H(j,i),H(j,i+1)); 
        
        i=2:n+3;
        j=2:n+2;
         
        Hy(j,i) = (Vpos(j,i)).*H(j,i) + (Vneg(j,i)).*H(j+1,i) ;%+ (V(j,i)==0).*max(H(j,i),H(j+1,i)); 
        
              
      %% Update U
       i =2:n+2;
       j= 3:n+2;
             
     
       U(j,i) = (U(j,i) - g*dt/dx* (H(j,i+1)-H(j,i))  ...                                %Gravity term
          +(Upos(j,i)).*(-   dt/dx* (U(j,i)-U(j,i-1))    .* (U(j,i)+U(j,i-1))/2) ...      %advection U+
          +(Vpos(j,i)).*(-   dt/dy* (U(j,i)-U(j-1,i))    .* (V(j-1,i)+V(j-1,i+1))/2) ...  %advection V+  
          +(Uneg(j,i)).*(-   dt/dx* (U(j,i+1)-U(j,i))    .* (U(j,i)+U(j,i+1))/2) ...      %advection U-
          +(Vneg(j,i)).*(-   dt/dy* (U(j+1,i)-U(j,i))    .* (V(j,i)+V(j,i+1))/2)  )...    %advection V-
          ./(1+dt*cf*sqrt((U(j,i).^2+((V(j,i)+V(j-1,i))/2).^2)) ./(H(j,i)+H(j,i+1)));    %Bottom friction (implicit)
      
      %boundary terms seperately
      j = n+3;
      U(j,i) = (U(j,i) - g*dt/dx* (H(j,i+1)-H(j,i))  ...                                %Gravity term
          +(Upos(j,i)).*(-   dt/dx* (U(j,i)-U(j,i-1))    .* (U(j,i)+U(j,i-1))/2) ...      %advection U+
                        -   dt/dy* (U(j,i)-U(j-1,i))    .* (V(j-1,i)+V(j-1,i+1))/2 ...  %advection V+  
          +(Uneg(j,i)).*(-   dt/dx* (U(j,i+1)-U(j,i))    .* (U(j,i)+U(j,i+1))/2)) ...      %advection U-
          ./(1+dt*cf*sqrt((U(j,i).^2+((V(j,i)+V(j-1,i))/2).^2)) ./(H(j,i)+H(j,i+1)));    %Bottom friction (implicit)
      
      j = 2;
      U(j,i) = (U(j,i) - g*dt/dx* (H(j,i+1)-H(j,i))  ...                                %Gravity term
          +(Upos(j,i)).*(-   dt/dx* (U(j,i)-U(j,i-1))    .* (U(j,i)+U(j,i-1))/2) ...      %advection U+   
          +(Uneg(j,i)).*(-   dt/dx* (U(j,i+1)-U(j,i))    .* (U(j,i)+U(j,i+1))/2) ...      %advection U-
                        -   dt/dy* (U(j+1,i)-U(j,i))    .* (V(j,i)+V(j,i+1))/2)  ...    %advection V-
          ./(1+dt*cf*sqrt((U(j,i).^2+((V(j,i)+V(j-1,i))/2).^2)) ./(H(j,i)+H(j,i+1)));    %Bottom friction (implicit)
      
      %update boolean arrays
       Upos=U>0;  Uneg=1-Upos;               
       %% Update V 
       i=3:n+2;
       j=2:n+2;
       
       V(j,i) =  (V(j,i) - g*dt/dy*   (H(j+1,i)-H(j,i))    ...                             %Gravity                             
              +(Vpos(j,i)).*(- dt/dy*  (V(j,i)-V(j-1,i))    .*(V(j,i)+V(j-1,i))/2) ...      %advection V+
              +(Upos(j,i)).*(- dt/dx*  (V(j,i)-V(j,i-1))    .*(U(j+1,i-1)+U(j,i-1))/2) ...  %advection U+   
              +(Vneg(j,i)).*(- dt/dy*( (V(j+1,i)-V(j,i))    .*(V(j,i)+V(j+1,i))/2)) ...     %advection V-
              +(Uneg(j,i)).*(- dt/dx*  (V(j,i+1)-V(j,i))    .*(U(j+1,i)+U(j,i))/2)) ...     %advection U-                                                                                 %advection V-
              ./(1+dt*cf*sqrt( ((U(j,i)+U(j,i-1))/2).^2+V(j,i).^2)  ./(H(j,i)+H(j+1,i)));  %Bottom friction (implicit) 
       
      %seperate boundaries
      i=2;    
      V(j,i) =  (V(j,i) - g*dt/dy*   (H(j+1,i)-H(j,i))    ...                             %Gravity                             
              +(Vpos(j,i)).*(- dt/dy*  (V(j,i)-V(j-1,i))    .*(V(j,i)+V(j-1,i))/2) ...      %advection V+   
              +(Vneg(j,i)).*(- dt/dy*( (V(j+1,i)-V(j,i))    .*(V(j,i)+V(j+1,i))/2)) ...     %advection V-
                            - dt/dx*  (V(j,i+1)-V(j,i))    .*(U(j+1,i)+U(j,i))/2) ...     %advection U-                                                                                 %advection V-
              ./(1+dt*cf*sqrt( ((U(j,i)+U(j,i-1))/2).^2+V(j,i).^2)  ./(H(j,i)+H(j+1,i)));  %Bottom friction (implicit)
     
      i=n+3;    
      V(j,i) =  (V(j,i) - g*dt/dy*   (H(j+1,i)-H(j,i))    ...                             %Gravity                             
              +(Vpos(j,i)).*(- dt/dy*  (V(j,i)-V(j-1,i))    .*(V(j,i)+V(j-1,i))/2) ...      %advection V+
                            - dt/dx*  (V(j,i)-V(j,i-1))    .*(U(j+1,i-1)+U(j,i-1))/2 ...  %advection U+   
              +(Vneg(j,i)).*(- dt/dy*( (V(j+1,i)-V(j,i))    .*(V(j,i)+V(j+1,i))/2)) ) ...     %advection V-                                                                                 %advection V-
              ./(1+dt*cf*sqrt( ((U(j,i)+U(j,i-1))/2).^2+V(j,i).^2)  ./(H(j,i)+H(j+1,i)));  %Bottom friction (implicit) 
    
   %update boolean arrays       
   Vpos = V>0; Vneg=1-Vpos;
        %% Update H
    i=2:n+3;
    j=2:n+3;

    H(j,i) = H(j,i) - dt/dx*(Hx(j,i).*U(j,i)-Hx(j,i-1).*U(j,i-1))...
                    - dt/dy*(Hy(j,i).*V(j,i)-Hy(j-1,i).*V(j-1,i));
       
   %% CFL Condition
      % Advection free CFL condition
      dt = min(tstep,safety*dx/(2*sqrt(2)*sqrt(g*max(max(H))))); %safety factor of .8
     %dt = .01;
      % CFL condition with velocity             
      %    dt = min(tstep,    ...
       %  safety*dx/max(max(  ...    %safety factor of .3
        %   max( 1*abs(U(j,i)+U(j-1,i)./(2*H(j,i)))    + sqrt(g*H(j,i)) ,         ...
         %       1*abs(V(j,i)+V(j,i-1)./(2*H(j,i)))   + sqrt(g*H(j,i)) ...
          % ))));
       
       
        %% Update plot
       if mod(nstep,nplotstep) == 0
        if timer == 1
        wait(G);       
         tc = toc;
       end
          C = abs(U(i,j)) + abs(V(i,j));  % Color shows momemtum
          t = nstep;
          tv = norm(C,'fro');
         % set(surfplot,'zdata',H(j,i),'cdata',C);
          %set(top,'string',sprintf('t = %6.2f,  tv = %6.2f',t,tv))  
          
            surf(xq/n,xq/n,H(xq,xq),C(xq,xq));
            grid off
            axis([0 1 0 1 0 2*Hstart])
            caxis([-1 1])
            %surfplot.EdgeColor = 'none';
            colormap default
            set(title(''),'string',sprintf('t = %6.2f,  tv = %6.2f, dt = %6.2f, tc = %6.2f',t,tv,dt,tc))
            drawnow
            
      tic
       end
     

   end
   
end
close(gcf)
end
  function D = droplet(height,width)
% DROPLET  2D Gaussian
% D = droplet(height,width)
   [x,y] = ndgrid(-1:(2/(width-1)):1);
   D = height*exp(-5*(x.^2+y.^2));
  end
% ------------------------------------
function [surfplot,top,restart,quit] = initgraphics(n,Hstart);
% INITGRAPHICS  Initialize graphics for waterwave.
% [surfplot,top,restart,quit] = initgraphics(n)
% returns handles to a surface plot, its title, and two uicontrol toggles.
   clf
   shg
   set(gcf,'menubar','none','numbertitle','off','name','Waterwave') 
   x = gpuArray((0:n-1)/(n-1));
   surfplot = surf(x,x,gpuArray(ones(n,n)),gpuArray(zeros(n,n)));
   grid off
   axis([0 1 0 1 Hstart-1 Hstart+2])
   caxis([-1 1])
   surfplot.EdgeColor = 'none';
   colormap default
   c = (1:64)'/64;
   %cyan = [0*c c c];
   %colormap(default)
   
   top = title('xxx');
   restart = uicontrol('position',[20 20 80 20],'style','toggle','string','restart');
   quit = uicontrol('position',[120 20 80 20],'style','toggle','string','close');
end
