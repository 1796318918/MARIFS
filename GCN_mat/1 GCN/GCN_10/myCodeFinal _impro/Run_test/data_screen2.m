load PPMI_scores_829_187NC_588PD_54SWEDD_noNaN_baseline;
dataT1_CSF_DTI=PPMI_scores_829_187NC_588PD_54SWEDD_baseline(:,2:end);
Is_zero_index=all(dataT1_CSF_DTI(:,1:116)==0,2)+all(dataT1_CSF_DTI(:,349:464)==0,2);
%PD
Is_PD=(dataT1_CSF_DTI(:,1977)==2);
Is_zero_PD=Is_zero_index+Is_PD;
index1=find(Is_zero_PD==2);

%SWEDD
Is_SWEDD=(dataT1_CSF_DTI(:,1977)==1);
Is_zero_SWEDD=Is_zero_index+Is_SWEDD;
index2=find(Is_zero_SWEDD==2);

%NC
Is_NC=(dataT1_CSF_DTI(:,1977)==0);
Is_zero_NC=Is_zero_index+Is_NC;
index3=find(Is_zero_NC==2);

data=PPMI_scores_829_187NC_588PD_54SWEDD_baseline;
data([index1;index2;index3],:)=[];
PPMI_scores_297_65NC_197PD_35SWEDD_baseline=data;