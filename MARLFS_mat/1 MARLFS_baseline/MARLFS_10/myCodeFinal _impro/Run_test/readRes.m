

% 获取所有的.mat文件

clear
resPath=cd;
allRes=dir([resPath,'\*.mat']);


len=length(allRes);


select='Res1';      % 选择最大Fscore
% select='Res2';      % 选择最大Acc

for i=1:len
    strName{i}=allRes(i).name;
    load (strName{i}); % 依次读取mat

    Res=eval(select); % 选择最大Acc
    
    a(i,1)=Res.svmAcc;
    a(i,2)=Res.svmSen;
    a(i,3)=Res.svmSpec;
    a(i,4)=Res.svmPrec;
    a(i,5)=Res.svmFscore;
    a(i,6)=Res.svmAuc;
    
    a(i,7)=Res.maxAcc;
    a(i,8)=Res.minArmse;
    
    a(i,9)=Res.maxBcc;
    a(i,10)=Res.minBrmse;
    
    a(i,11)=Res.maxCcc;
    a(i,12)=Res.minCrmse;
    
    a(i,13)=Res.maxDcc;
    a(i,14)=Res.minDrmse;
    
    a(i,15)=Res.svmAcc_Std;
    a(i,16)=Res.svmSen_Std;
    a(i,17)=Res.svmSpec_Std;
    a(i,18)=Res.svmPrec_Std;
    a(i,19)=Res.svmFscore_Std;
    a(i,20)=Res.svmAuc_Std;
    
    a(i,21)=Res.maxAcc_Std;
    a(i,22)=Res.minArmse_Std;
    a(i,23)=Res.maxBcc_Std;
    a(i,24)=Res.minBrmse_Std;
    a(i,25)=Res.maxCcc_Std;
    a(i,26)=Res.minCrmse_Std;
    a(i,27)=Res.maxDcc_Std;
    a(i,28)=Res.minDrmse_Std;

end


a=real(a);
matName=strName';
titleName={'Feature','ACC','SEN','SPEC','PREC','Fscore','AUC','scoreAcc','ARMSE','scoreBcc','BRMSE','scoreCcc','CRMSE','scoreDcc','DRMSE'};
titleName2={'ACC_std','SEN_std','SPEC_std','PREC_std','Fscore_std','AUC_std','scoreAcc_std','ARMSE_std','scoreBcc_std','BRMSE_std','scoreCcc_std','CRMSE_std','scoreDcc_std','DRMSE_std'};
combine=[matName,num2cell(a)];

titleName=[titleName,titleName2];
combine=[titleName;combine];


xlsFile=['E:\brain processing\Task_Explain\PD_Code\Res\',select,'.xlsx'];
xlswrite(xlsFile, combine); % 将need_xml_info直接转换成xls格式




