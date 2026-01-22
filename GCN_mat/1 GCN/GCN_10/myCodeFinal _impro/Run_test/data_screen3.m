load PPMI_scores_297_65NC_197PD_35SWEDD_baseline;
dataT1_CSF_DTI=PPMI_scores_297_65NC_197PD_35SWEDD_baseline(:,2:end);
% Is_zero_index=all(dataT1_CSF_DTI(:,1:116)==0,2)+all(dataT1_CSF_DTI(:,349:464)==0,2);
%PD
Is_PD=(dataT1_CSF_DTI(:,1977)==2);
% Is_zero_PD=(~Is_zero_index)+Is_PD;
index1=find(Is_PD==1);

data=PPMI_scores_297_65NC_197PD_35SWEDD_baseline;
data(index1(end-40:end),:)=[];
PPMI_scores_256_65NC_156PD_35SWEDD_baseline=data;