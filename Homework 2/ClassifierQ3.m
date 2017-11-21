function ClassifierQ2(Data,Name)

% Arranging Data
TrueLabels_DataTr=[Data(1:60,4);Data(101:160,4)];
TrueLabels_DataVal=[Data(61:100,4);Data(161:end,4)];
DataTr=[Data(1:60,1:3);Data(101:160,1:3)];
DataTr_Class0=DataTr(1:60,:);
DataTr_Class1=DataTr(61:120,:);
DataVal=[Data(61:100,1:3);Data(161:end,1:3)];
N_tr=60; N_val=40;

% LDA
DataTr0Mean=mean(DataTr_Class0);
Class0=DataTr_Class0-DataTr0Mean;
Cov_DataTr0=(Class0'*Class0);
DataTr1Mean=mean(DataTr_Class1);
Class1=DataTr_Class1-DataTr1Mean;
Cov_DataTr1=(Class1'*Class1);
Cov_DataTr=Cov_DataTr0+Cov_DataTr1;
w=Cov_DataTr\(DataTr0Mean-DataTr1Mean)';
w=w/norm(w);
ProjectedDataTr=w'*(DataTr)';
ProjectedDataVal=w'*(DataVal)';

figure; 
hAxes = axes('NextPlot','add',...           
             'DataAspectRatio',[1 1 1],...  
             'YLim',[-1.6 1.6],...             
             'Color','white');              
scatter(ProjectedDataTr(1:60),zeros(1,60),25,[1 0.5469 0],'o','DisplayName','C_0')
scatter(ProjectedDataTr(61:120),zeros(1,60),25,[0.1172 0.5625 1],'o','DisplayName','C_1')
yticks([])
xlabel('d')
legend('Show','Location','northeast')
title(['Visualization of Projected Training Dataset ',Name,' onto One Dimension'])

figure; 
hAxes = axes('NextPlot','add',...           
             'DataAspectRatio',[1 1 1],...  
             'YLim',[-1.6 1.6],...             
             'Color','white');               
scatter(ProjectedDataVal(1:40),zeros(1,40),25,[1 0.5469 0],'x','DisplayName','C_0')
scatter(ProjectedDataVal(41:80),zeros(1,40),25,[0.1172 0.5625 1],'x','DisplayName','C_1')
yticks([])
xlabel('d')
legend('Show','Location','northeast')
title(['Visualization of Projected Validation Dataset ',Name,' onto One Dimension'])

mean0=mean(ProjectedDataTr(1:60));
mean1=mean(ProjectedDataTr(61:120));
std0=std(ProjectedDataTr(1:60));
std1=std(ProjectedDataTr(61:120));

% Predicting Classes for Data

PredictedLabels_DataTr=zeros(120,1);
PredictedLabels_DataVal=zeros(80,1);
for i=1:(2*N_tr)
    g1_tr=-log(std1)-((ProjectedDataTr(i)-mean1)^2)/(2*std1^2);
    g0_tr=-log(std0)-((ProjectedDataTr(i)-mean0)^2)/(2*std0^2);
    if g1_tr>g0_tr
        PredictedLabels_DataTr(i)=1;
    end
end

for i=1:(2*N_val)
    g1_val=-log(std1)-((ProjectedDataVal(i)-mean1)^2)/(2*std1^2);
    g0_val=-log(std0)-((ProjectedDataVal(i)-mean0)^2)/(2*std0^2);
    if g1_val>g0_val
        PredictedLabels_DataVal(i)=1;
    end
end

TrConfusion=confusionmat(TrueLabels_DataTr,PredictedLabels_DataTr);
ValConfusion=confusionmat(TrueLabels_DataVal,PredictedLabels_DataVal);
disp('TrConfusion: ')   
disp(TrConfusion)
disp('ValConfusion: ')   
disp(ValConfusion)

end