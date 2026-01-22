

function RunFunProposed_SVR2_lambda_4_add_age_sex(num1,num2,meanSelect,svmParc,svmParg,regScoreSelect,isYscoresCentering)

% num1; % 选择三种分组组合之一   NCvsPD NCvsSWEDD PDvsSWEDD
% num2; % 选择八种特征组合之一
% meanSelect 均值选择阈值
% svmParc,svmParg SVM参数列表
% regScoreSelect：选择谁作为【回归变量】，同时预测DSSM四个向量，回归变量会在RunFunction2_impro中自动加上“label”.
    % depScores sleepScores smellScores MoCAScores CSF1 CSF2 CSF3
    % regScoreSelect={'depScores','MoCAScores','CSF2'}; 
    % regScoreSelect={'depScores','sleepScores','smellScores','MoCAScores'};  
% isYscoresCentering,回归变量是否中心化


kFold=1;	% 对样本进行 kFold 次划分
subFold=10;	% 每次划分的子集个数
% 

% t1=-2; parsLambda1 = 10.^t1;
% t2=10;  parsLambda2 = t2; 
% t3=0;  parsLambda3 = 10.^t3;
% % 特征选择
% t4=0;  parsLambda4 = t4;

t1=1; parsLambda1 = t1;
t2=1;  parsLambda2 = t2; 
t3=1;  parsLambda3 = t3;
t4=1;  parsLambda4 = t4;


parsIte=50; % 迭代次数

combineSelect1=num1; % 选择三种分组组合之一   NCvsPD NCvsSWEDD PDvsSWEDD
combineSelect2=num2; % 选择八种特征组合之一

%hzw 2022.9.5 
% 选择不同组进行实验 NCvsPD NCvsSWEDD PDvsSWEDD
groupSelectCell{1}={'isPD','isNC','PD','NC'};          
groupSelectCell{2}={'isSWEDD','isNC','SWEDD','NC'};
groupSelectCell{3}={'isPD','isSWEDD','PD','SWEDD'};

% 选择哪些特征值
% X1：灰质   % X2：白质    % M:T1CSF   % Y：FA   %Z:MD    %L1 L2 L3    %V1 V2 V3
% A:附加数据（4个得分+age+sex）
mergeDataSelectCell{1}={'X1'};              % GM  
mergeDataSelectCell{2}={'X2'};              % WM
mergeDataSelectCell{3}={'M'};               % CSF
mergeDataSelectCell{4}={'Y'};               % FA
mergeDataSelectCell{5}={'Z'};               % MD
mergeDataSelectCell{6}={'L1'};              % L1
mergeDataSelectCell{7}={'L2'};              % L2
mergeDataSelectCell{8}={'L3'};              % L3
mergeDataSelectCell{9}={'V1'};              % V1
mergeDataSelectCell{10}={'V2'};             % V2
mergeDataSelectCell{11}={'V3'};             % V3

mergeDataSelectCell{11}={'X1','L1'};               % DTI_FA
mergeDataSelectCell{12}={'X1','L1','V1_1'};               % DTI_FA
mergeDataSelectCell{13}={'X1','L1','SP'};               % hzw 2024.11.8
mergeDataSelectCell{14}={'X1','L1','V1_1','SP'};               % hzw 2024.11.8

groupSelect=groupSelectCell{combineSelect1};
mergeDataSelect=mergeDataSelectCell{combineSelect2};

regScoreShortName=[];

depSelect=0;
sleepSelect=0;
smellSelect=0;
MoCASelect=0;
    
% DSlSmMc1c2c3：Sl表示sleep Sm表示smell  c1表示AB42 c2表示tTau c3表示pTau
if(ismember('depScores',regScoreSelect))
    regScoreShortName=[regScoreShortName,'D']; 
    depSelect=1;
end
if(ismember('sleepScores',regScoreSelect))
    regScoreShortName=[regScoreShortName,'Sl'];  
    sleepSelect=1;
end
if(ismember('smellScores',regScoreSelect))
    regScoreShortName=[regScoreShortName,'Sm']; 
    smellSelect=1;
end
if(ismember('MoCAScores',regScoreSelect))
    regScoreShortName=[regScoreShortName,'M'];    
    MoCASelect=1;
end



% 保存的mat名字
matNameCell1={{'PDvsNC'},{'SWvsNC'},{'PDvsSW'}};
% matNameCell2={{'T1G'},{'T1C'},{'DTI'},{'GCD'},{'GCDF'}}; 
% matNameCell2={{'T1G'},{'T1C'},{'DTI'},{'T1G_T1C_DTI'},{'T1G_T1C_DTI_CSF'},{'T1G_T1C'},{'T1G_DTI'},{'T1C_DTI'}};  


regSelect=1;  % 中心化
isCentring='regCentering';
if(isYscoresCentering==0)
    isCentring='noRegCenter';
    regSelect=0; % 不中心化
end


matNameIndexPart1=num2str(regSelect);           % 是否中心化
matNameIndexPart2=[num2str(depSelect),num2str(sleepSelect),num2str(smellSelect),num2str(MoCASelect)];   % DSSM组合
% matNameIndexPart3=[num2str(C1Select),num2str(C2Select),num2str(C3Select)];  % C1,C2,C3,还是C1C2C3
matNameIndexPart3=num2str(combineSelect1);   % 是否中心化
matNameIndexPart4=num2str(combineSelect2);   % 是否中心化


matName=[...
    'R',matNameIndexPart1,'_',...
    'D',matNameIndexPart2,'_',...
    matNameIndexPart3,'.',...
    matNameIndexPart4,'.',...
    'Mat_',...
    '_',matNameCell1{combineSelect1}{1},...%,'_',matNameCell2{combineSelect2}{1},...
    '_','Proposed',...    
    '_',regScoreShortName,...
    '_',isCentring,...
    '_','mean[',num2str(meanSelect*100),']',...
    '_','c[',num2str(min(svmParc)),num2str(max(svmParc)),']',...
    '_','g[',num2str(min(svmParg)),num2str(max(svmParg)),']',...
    '_','p1[',num2str(min(t1)),num2str(max(t1)),']',...
    '_','p2[',num2str(min(t2)),num2str(max(t2)),']',...
    '_','p3[',num2str(min(t3)),num2str(max(t3)),']',...
    '_','p4[',num2str(min(t4)),num2str(max(t4)),']',...
    '_','Ite[',num2str(parsIte),']',...
    '_','SVR2_baseline',...
    '.mat'];


RunFunction2_impro_SVR2_lambda_4_add_age_sex(groupSelect,mergeDataSelect,regScoreSelect,kFold,subFold,svmParc,svmParg,parsLambda1,parsLambda2,parsLambda3,parsLambda4,parsIte,meanSelect,matName,isYscoresCentering);

end



