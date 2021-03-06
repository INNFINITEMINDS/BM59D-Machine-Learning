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
%Evidence
Evidence=Class_1_Likelihood*Class_1_Prior+Class_2_Likelihood*Class_2_Prior;
%% SECOND PART
%Class Posteriors
Class_1_Posterior=(Class_1_Prior*Class_1_Likelihood)./Evidence;
Class_2_Posterior=(Class_2_Prior*Class_2_Likelihood)./Evidence;
%Risks
Class_1_Risk=0.5*(1-Class_1_Posterior);
Class_2_Risk=1-Class_2_Posterior;
Reject_Risk=ones(1,length(x))*0.2;
figure
p1=plot(x,Class_1_Risk)
hold on
p2=plot(x,Class_2_Risk)
p3=plot(x,Reject_Risk)
Boundary = get(gca,'YLim');
line([-14.825 -14.825],Boundary,'Color','k','LineStyle','--');
line([-10.305 -10.305],Boundary,'Color','k','LineStyle','--');
line([2.275 2.275],Boundary,'Color','k','LineStyle','--');
line([6.795 6.795],Boundary,'Color','k','LineStyle','--');
xlim([-25 17])
xlabel('x')
ylabel('R(\alpha_i|x)')
title('Class Risks with \lambda_1_2=0.5, \lambda_2_1=1 and \lambda_3=0.2')
legend([p1 p2 p3], 'R_1','R_2','R_r_e_j_e_c_t')

%% DECISION RULES and CONFUSION MATRICES
% For minimum risk, C_2   x<-14.82
%                   R     -14.83<x<-10.30
%                   C_1   -10.31<x<2.28
%                   R     2.27<x<6.8
%                   C_2   x>6.79

TrClassification=zeros(125,1);
ValClassification=zeros(125,1);
for i=1:125
    if xtr_classification(i,1)<-14.82
        TrClassification(i)=2;
    elseif xtr_classification(i,1)<-10.30
        TrClassification(i)=3;
    elseif xtr_classification(i,1)<2.28
        TrClassification(i)=1;
    elseif xtr_classification(i,1)<6.8
        TrClassification(i)=3;
    else
        TrClassification(i)=2;
    end
    
    if xval_classification(i,1)<-14.82
        ValClassification(i)=2;
    elseif xval_classification(i,1)<-10.30
        ValClassification(i)=3;
    elseif xval_classification(i,1)<2.28
        ValClassification(i)=1;
    elseif xval_classification(i,1)<6.8
        ValClassification(i)=3;
    else
        ValClassification(i)=2;
    end
end

TrConfusion=confusionmat(xtr_classification(:,2),TrClassification)
ValConfusion=confusionmat(xval_classification(:,2),ValClassification)



