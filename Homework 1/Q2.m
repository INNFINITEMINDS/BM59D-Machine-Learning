%% LOAD AND ASSIGN DATA
clc;
clear all;
load('Data.mat');
%% POLYNOMIAL FITTING AND ERRORS
x=x_regression(:,4);
y=x_regression(:,1:3);
y_actual=x.^3-x+1;

Associated_Error=zeros(10,3);
Actual_Error=zeros(10,3);

Theta_Mat={zeros(10,10),zeros(10,10),zeros(10,10)};
Fitted_Poly={zeros(10,30),zeros(10,30),zeros(10,30)};

for i=1:3
    X=ones(30,1);
    for j=1:10
        Theta=(((X'*X)\(X'))*y(:,i))';
        Theta_Mat{i}(j,1:length(Theta))=Theta;
        Fitted_Poly{i}(j,:)=Theta*X';
        X=[X x.^j];
        Associated_Error(j,i)=mean((y(:,i)-(Fitted_Poly{i}(j,:))').^2);
        Actual_Error(j,i)=mean((y_actual-(Fitted_Poly{i}(j,:))').^2);
    end
end

%% PLOTTING ERRORS

for i=1:3
    figure;
    plot(0:9,Associated_Error(:,i),'DisplayName','Noisy Error')
    hold on 
    plot(0:9,Actual_Error(:,i),'DisplayName','Actual Error')
    xlabel('Order of Polynomial')
    ylabel('Error')
    title(['Error with Noisy Data (\sigma_n=' num2str(0.5-(i-1)*0.2) ') vs. Actual Data'])
    legend('show')
end

%% PLOTTING FITTED CURVES 

for i=1:3
    figure
    hold on
    plot(x,Fitted_Poly{i}(3,:),'','DisplayName','2^{nd} Order')
    plot(x,Fitted_Poly{i}(4,:),'DisplayName','3^r^d Order')
    plot(x,Fitted_Poly{i}(7,:),'DisplayName','6^t^h Order')
    plot(x,Fitted_Poly{i}(10,:),'DisplayName','9^t^h Order')
    scatter(x,y_actual,'k','x','DisplayName','Actual Data')
    xlabel('x')
    ylabel('f(x)')
    title(['Polynomial Fittings (\sigma_n=' num2str(0.5-(i-1)*0.2) ') vs. Actual Data'])
    legend('show')
    axis tight
end

for i=1:3
    figure
    hold on
    plot(x,Fitted_Poly{i}(3,:),'','DisplayName','2^n^d Order')
    plot(x,Fitted_Poly{i}(4,:),'DisplayName','3^r^d Order')
    plot(x,Fitted_Poly{i}(7,:),'DisplayName','6^t^h Order')
    plot(x,Fitted_Poly{i}(10,:),'DisplayName','9^t^h Order')
    plot(x,y(:,i),'k--','DisplayName','Noisy Data')
    xlabel('x')
    ylabel('f(x)')
    title(['Polynomial Fittings vs. Noisy Data (\sigma_n=' num2str(0.5-(i-1)*0.2) ')'])
    legend('show')
    axis tight
end

