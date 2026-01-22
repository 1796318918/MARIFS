[num,txt,raw]=xlsread('PPMI_scores_846_190NC_600PD_56SWEDD_baseline.csv');
data=num;
delete_index=isnan(data(:,end-4:end-1));
data(sum(delete_index,2)~=0,:)=[];
PPMI_scores_829_187NC_588PD_54SWEDD_baseline=data;