

addpath(genpath('C:\Users\hantao19921227\Desktop\huang zhongwei\libsvm-mat-2.91-1'))
addpath(genpath('..\'));
addpath(genpath('..\..\'));


meanSelect=0.2;
svmParc=-10:1:10; % libsvm的参数设置：2的-5次方—>2的5次方
svmParg=-5:1:5;


% depScores sleepScores smellScores MoCAScores CSF1 CSF2 CSF3


% DSlSmM 2种组合
regScoreSelect={'sleepScores','smellScores'};  % DSm D0110

csfSelect{4}={''};

finalSelect=[regScoreSelect,csfSelect{4}];


tic
sTime=toc;  % 开始时间

k=1; % 中心化

% NC vs PD 
% RunFunProposed_SVR2_lambda_4(1,1,meanSelect,svmParc,svmParg,finalSelect,k)
% eTime1=toc; fprintf('RunFunProposed_SVR2(1,1)耗时：%fs\n',eTime1-sTime);
% 
% RunFunProposed_SVR2_lambda_4(1,2,meanSelect,svmParc,svmParg,finalSelect,k)
% eTime2=toc; fprintf('RunFunProposed_SVR2(1,2)耗时：%fs\n',eTime2-eTime1);
% 
% RunFunProposed_SVR2_lambda_4(1,3,meanSelect,svmParc,svmParg,finalSelect,k)
% RunFunProposed_SVR2_lambda_4(1,4,meanSelect,svmParc,svmParg,finalSelect,k)

% % % NC vs SW 
% RunFunProposed_SVR2_lambda_4(2,1,meanSelect,svmParc,svmParg,finalSelect,k)
% eTime3=toc; fprintf('RunFunProposed_SVR2(2,1)耗时：%fs\n',eTime3-eTime2);
% 
% RunFunProposed_SVR2_lambda_4(2,2,meanSelect,svmParc,svmParg,finalSelect,k)
% eTime4=toc; fprintf('RunFunProposed_SVR2(2,2)耗时：%fs\n',eTime4-eTime3);

RunFunProposed_SVR2_lambda_4(2,3,meanSelect,svmParc,svmParg,finalSelect,k)
eTime5=toc; fprintf('RunFunProposed_SVR2(2,3)耗时：%fs\n',eTime5-eTime4);

RunFunProposed_SVR2_lambda_4(2,4,meanSelect,svmParc,svmParg,finalSelect,k)
% eTime6=toc; fprintf('RunFunProposed_SVR2(2,4)耗时：%fs\n',eTime6-eTime5);

eTime=toc;  % 结束时间
fprintf('总共耗时：%fs\n',eTime-sTime);




