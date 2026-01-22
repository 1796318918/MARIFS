

% addpath(genpath('C:\Users\hantao19921227\Desktop\huang zhongwei\libsvm-mat-2.91-1'))
addpath(genpath('../'));
% addpath(genpath('..\..\'));


meanSelect=0.2;
svmParc=-10:1:10; % libsvm的参数设置：2的-5次方—>2的5次方
svmParg=-5:1:5;


% depScores sleepScores smellScores MoCAScores CSF1 CSF2 CSF3


% DSlSmM 2种组合
regScoreSelect={'depScores','sleepScores','smellScores','MoCAScores'};  % DSm D0110

csfSelect{4}={''};

finalSelect=[regScoreSelect,csfSelect{4}];


tic
sTime=toc;  % 开始时间

k=1; % 中心化

% hzw 2024.11.8
RunFunProposed_SVR2_lambda_4_add_age_sex(1,14,meanSelect,svmParc,svmParg,finalSelect,k);
RunFunProposed_SVR2_lambda_4_add_age_sex(2,14,meanSelect,svmParc,svmParg,finalSelect,k);
RunFunProposed_SVR2_lambda_4_add_age_sex(3,14,meanSelect,svmParc,svmParg,finalSelect,k);

eTime=toc;  % 结束时间
fprintf('总共耗时：%fs\n',eTime-sTime);




