%% LOAD DATA
clc;
clear all;
load('Data.mat');
p1=xtr_classification(:,2);
p1(p1<0)=2;
xtr_classification(:,2)=p1;
p2=xval_classification(:,2);
p2(p2<0)=2;
xval_classification(:,2)=p2;
%% FIRST PART
%Class Priors
n=length(xtr_classification(:,2)); 
n1=sum(xtr_classification(:,2)==1); 
Class_1_Prior=n1/n;
Class_2_Prior=(n-n1)/n;
%Class Likelihoods
Class_1=xtr_classification(1:n1,1);
Class_2=xtr_classification(n1+1:end,1);
mean1=mean(Class_1); std1=std(Class_1);
mean2=mean(Class_2); std2=std(Class_2);
x=-30:0.01:50;
Class_1_Likelihood=(1/(std1*sqrt(2*pi)))*exp((-(x-mean1).^2)/(2*std1^2));
Class_2_Likelihood=(1/(std2*sqrt(2*pi)))*exp((-(x-mean2).^2)/(2*std2^2));
figure
plot(x,Class_1_Likelihood,'DisplayName','C_1')
hold on
plot(x,Class_2_Likelihood,'DisplayName','C_2')
xlim([-30 50])
xlabel('x')
ylabel('p(x|C_i)')
title('Class Likelihoods')
legend('show')
%Evidence
Evidence=Class_1_Likelihood*Class_1_Prior+Class_2_Likelihood*Class_2_Prior;
figure
plot(x,Evidence)
xlabel('x')
ylabel('p(x)')
title('Evidence')
%% SECOND PART
%Class Posteriors
Class_1_Posterior=(Class_1_Prior*Class_1_Likelihood)./Evidence;
Class_2_Posterior=(Class_2_Prior*Class_2_Likelihood)./Evidence;
figure
p1=plot(x,Class_1_Posterior)
hold on
p2=plot(x,Class_2_Posterior)
Boundary = get(gca,'YLim');
line([-11.565 -11.565],Boundary,'Color','k','LineStyle','--');
line([3.535 3.535],Boundary,'Color','k','LineStyle','--');
xlim([-25 17])
xlabel('x')
ylabel('P(C_i|x)')
title('Class Posteriors')
legend([p1 p2], 'C_1','C_2')
%Risks
Class_1_Risk=1-Class_1_Posterior;
Class_2_Risk=1-Class_2_Posterior;
figure
p3=plot(x,Class_1_Risk)
hold on
p4=plot(x,Class_2_Risk)
Boundary = get(gca,'YLim');
line([-11.565 -11.565],Boundary,'Color','k','LineStyle','--');
line([3.535 3.535],Boundary,'Color','k','LineStyle','--');
xlim([-25 17])
xlabel('x')
ylabel('R(\alpha_i|x)')
title('Class Risks')
legend([p3 p4], 'R_1','R_2')
%Discriminants
Class_1_Discriminant=log(Class_1_Likelihood)+log(Class_1_Prior);
Class_2_Discriminant=log(Class_2_Likelihood)+log(Class_2_Prior);
figure
p5=plot(x,Class_1_Discriminant)
hold on
p6=plot(x,Class_2_Discriminant)
Boundary = get(gca,'YLim');
line([-11.565 -11.565],Boundary,'Color','k','LineStyle','--');
line([3.535 3.535],Boundary,'Color','k','LineStyle','--');
xlim([-25 17])
ylim([-25 0])
xlabel('x')
ylabel('g_i(x)')
title('Class Discriminants')
legend([p5 p6], 'g_1','g_2')

%% DECISION RULES and CONFUSION MATRICES
% For minimum risk, C_1   -11.57<x<3.54
%                   C_2   else

TrClassification=zeros(125,1);
ValClassification=zeros(125,1);
for i=1:125
    if xtr_classification(i,1)>-11.57 && xtr_classification(i,1)<3.54
        TrClassification(i)=1;
    else
        TrClassification(i)=2;
    end
    
    if xval_classification(i,1)>-11.57 && xval_classification(i,1)<3.54
        ValClassification(i)=1;
    else
        ValClassification(i)=2;
    end
end

TrConfusion=confusionmat(xtr_classification(:,2),TrClassification)
ValConfusion=confusionmat(xval_classification(:,2),ValClassification)
