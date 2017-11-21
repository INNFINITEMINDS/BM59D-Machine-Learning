function [TrConfusion,ValConfusion] = ClassifierQ1(Data,Name,Part)

% Arranging Data
TrueLabels_DataTr=[Data(1:60,4);Data(101:160,4)];
TrueLabels_DataVal=[Data(61:100,4);Data(161:end,4)];
DataTr=[Data(1:60,1:3);Data(101:160,1:3)];
DataVal=[Data(61:100,1:3);Data(161:end,1:3)];
N_tr=60; N_val=40; d=3;

% Visualization of Data
figure;  hold on; grid on;
scatter3(DataTr(1:60,1),DataTr(1:60,2),DataTr(1:60,3),25,[1 0.5469 0],'o','DisplayName','Training C_0')
scatter3(DataVal(1:40,1),DataVal(1:40,2),DataVal(1:40,3),25,[1 0.5469 0],'x','DisplayName','Validation C_0')
scatter3(DataTr(61:120,1),DataTr(61:120,2),DataTr(61:120,3),25,[0.1172 0.5625 1],'o','DisplayName','Training C_1')
scatter3(DataVal(41:80,1),DataVal(41:80,2),DataVal(41:80,3),25,[0.1172 0.5625 1],'x','DisplayName','Validation C_1')
view(-50,10)
legend('Show'); xlabel('d_1'); ylabel('d_2'); zlabel('d_3'); 
title(['Visualization of Training and Validation Dataset ',Name]);

% Bayesian Calculations for Data
DataMean_Class0=mean(DataTr(1:60,:));
DataMean_Class1=mean(DataTr(61:120,:));
Class_0=DataTr(1:60,:)-DataMean_Class0;
Class_1=DataTr(61:120,:)-DataMean_Class1;
Cov_Class0=(Class_0'*Class_0)./(N_tr);
Cov_Class1=(Class_1'*Class_1)./(N_tr);

if Part=='a'
    DataCov_Class1=Cov_Class1;
    DataCov_Class0=Cov_Class0;
end
if Part=='b'
    DataCov_Class0=0.5*Cov_Class0+0.5*Cov_Class1;
    DataCov_Class1=DataCov_Class0;
end
if Part=='c'
    DataCov_Class0=diag(diag(0.5*Cov_Class0+0.5*Cov_Class1));
    DataCov_Class1=DataCov_Class0;
end
if Part=='d'
    DataCov_Class0=diag([1 1 1])*mean(diag(0.5*Cov_Class0+0.5*Cov_Class1));
    DataCov_Class1=DataCov_Class0;
end

% Multivariate Gaussians for Class Likelihoods
s=0.2; x=(-12:s:23)';  y=(-13:s:18)'; 
z=(-10:s:15)'; [Y,X,Z]=meshgrid(y,x,z);
Class_0_Likelihood=zeros(length(x),length(y),length(z));
Class_1_Likelihood=zeros(length(x),length(y),length(z));

for i=1:length(x)
    for j=1:length(y)
        for k=1:length(z)
            Class_0_Likelihood(i,j,k)=(1/(sqrt(det(DataCov_Class0)*((2*pi)^d))))*exp((-0.5)*([x(i) y(j) z(k)]-DataMean_Class0)*(DataCov_Class0\([x(i) y(j) z(k)]-DataMean_Class0)'));
            Class_1_Likelihood(i,j,k)=(1/(sqrt(det(DataCov_Class1)*((2*pi)^d))))*exp((-0.5)*([x(i) y(j) z(k)]-DataMean_Class1)*(DataCov_Class1\([x(i) y(j) z(k)]-DataMean_Class1)'));
        end
    end
end

Isoprobability Surfaces for Data
figure; hold on; grid on;
isovalue = 0.82*(max(Class_0_Likelihood(:))-min(Class_0_Likelihood(:)))+min(Class_0_Likelihood(:));
surf1=isosurface(X,Y,Z,Class_0_Likelihood,isovalue);
p1 = patch(surf1);
set(p1,'FaceColor','red','EdgeColor','none','FaceAlpha',0.2);

isovalue = 0.87*(max(Class_0_Likelihood(:))-min(Class_0_Likelihood(:)))+min(Class_0_Likelihood(:));
surf2=isosurface(X,Y,Z,Class_0_Likelihood,isovalue);
p2 = patch(surf2);
set(p2,'FaceColor','yellow','EdgeColor','none','FaceAlpha',0.4);

isovalue = 0.92*(max(Class_0_Likelihood(:))-min(Class_0_Likelihood(:)))+min(Class_0_Likelihood(:));
surf3=isosurface(X,Y,Z,Class_0_Likelihood,isovalue);
p3 = patch(surf3);
set(p3,'FaceColor','cyan','EdgeColor','none','FaceAlpha',0.6);

isovalue = 0.97*(max(Class_0_Likelihood(:))-min(Class_0_Likelihood(:)))+min(Class_0_Likelihood(:));
surf4=isosurface(X,Y,Z,Class_0_Likelihood,isovalue);
p4 = patch(surf4);
set(p4,'FaceColor','blue','EdgeColor','none','FaceAlpha',1);

isovalue = 0.82*(max(Class_1_Likelihood(:))-min(Class_1_Likelihood(:)))+min(Class_1_Likelihood(:));
surf1=isosurface(X,Y,Z,Class_1_Likelihood,isovalue);
p1 = patch(surf1);
set(p1,'FaceColor','red','EdgeColor','none','FaceAlpha',0.2);

isovalue = 0.87*(max(Class_1_Likelihood(:))-min(Class_1_Likelihood(:)))+min(Class_1_Likelihood(:));
surf2=isosurface(X,Y,Z,Class_1_Likelihood,isovalue);
p2 = patch(surf2);
set(p2,'FaceColor','yellow','EdgeColor','none','FaceAlpha',0.4);

isovalue = 0.92*(max(Class_1_Likelihood(:))-min(Class_1_Likelihood(:)))+min(Class_1_Likelihood(:));
surf3=isosurface(X,Y,Z,Class_1_Likelihood,isovalue);
p3 = patch(surf3);
set(p3,'FaceColor','cyan','EdgeColor','none','FaceAlpha',0.6);

isovalue = 0.97*(max(Class_1_Likelihood(:))-min(Class_1_Likelihood(:)))+min(Class_1_Likelihood(:));
surf4=isosurface(X,Y,Z,Class_1_Likelihood,isovalue);
p4 = patch(surf4);
set(p4,'FaceColor','blue','EdgeColor','none','FaceAlpha',1);

ylim([-15 20])
zlim([-5 15])
xlim([-12 20])
xlabel('d_1'); ylabel('d_2'); zlabel('d_3')
title(['Multivariate Gaussian Likelihood Isoprobability Surfaces of Training Data ',Name])

% Predicting Classes for Data
PredictedLabels_DataTr=zeros(120,1);
PredictedLabels_DataVal=zeros(80,1);
for i=1:(2*N_tr)
    g1_tr=-0.5*(log(det(DataCov_Class1))+(DataTr(i,:)-DataMean_Class1)*(DataCov_Class1\(DataTr(i,:)-DataMean_Class1)'));
    g0_tr=-0.5*(log(det(DataCov_Class0))+(DataTr(i,:)-DataMean_Class0)*(DataCov_Class0\(DataTr(i,:)-DataMean_Class0)'));
    if g1_tr>g0_tr
        PredictedLabels_DataTr(i)=1;
    end
end

for i=1:(2*N_val)
    g1_val=-0.5*(log(det(DataCov_Class1))+(DataVal(i,:)-DataMean_Class1)*(DataCov_Class1\(DataVal(i,:)-DataMean_Class1)'));
    g0_val=-0.5*(log(det(DataCov_Class0))+(DataVal(i,:)-DataMean_Class0)*(DataCov_Class0\(DataVal(i,:)-DataMean_Class0)'));
    if g1_val>g0_val
        PredictedLabels_DataVal(i)=1;
    end
end

TrConfusion=confusionmat(TrueLabels_DataTr,PredictedLabels_DataTr)
ValConfusion=confusionmat(TrueLabels_DataVal,PredictedLabels_DataVal)

end