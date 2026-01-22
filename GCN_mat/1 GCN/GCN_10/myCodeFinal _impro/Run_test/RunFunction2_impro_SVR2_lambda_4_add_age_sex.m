

% RunFunction2 add scoresGroup  and preScoresGroup
function RunFunction2_impro_SVR2_lambda_4_add_age_sex(groupSelect,mergeDataSelect,regScoreSelect,kFold,subFold,svmParc,svmParg,parsLambda1,parsLambda2,parsLambda3,parsLambda4,parsIte,meanSelect,matName,isYscoresCentering)

% 有特征选取

% groupSelect: NCvsPD NCvsSWEDD PDvsSWEDD
%   例如：groupSelect={'isNC','isPD','NC','PD'}; 

% mergeDataSlect：[X1,X2,Y,Z,M]自由组合      % X1：灰质   X2：白质   %Y：DTI    %Z生物CSF    %M:T1CSF
%   例如:mergeDataSlect={'X1','Z'};

% kFold： 对样本进行 kFold 次划分
% subFold：每次划分的子集个数
% svmParc： libsvm的参数设置：2的-5次方—>2的5次方
% svmParg： libsvm的参数设置：2的-5次方—>2的5次方



disp(groupSelect);
disp(mergeDataSelect);
disp(regScoreSelect);
disp({'isYscoresCentering=',num2str(isYscoresCentering)})


% hzw 2024.11.8;
% load PPMI_scores_256_65NC_156PD_35SWEDD_baseline;
% dataT1_CSF_DTI=PPMI_scores_256_65NC_156PD_35SWEDD_baseline(:,2:end);
load PPMI_scores_256_65NC_156PD_35SWEDD_baseline_SPECT;
dataT1_CSF_DTI=PPMI_scores_256_65NC_156PD_35SWEDD_baseline_SPECT(:,2:end);
% dataT1_CSF_DTI=cell2mat(dataT1_CSF_DTI);


T1_G_116=dataT1_CSF_DTI(:,1:116);   % 116 灰质
T1_W_116=dataT1_CSF_DTI(:,117:232); % 116 白质
T1_C_116=dataT1_CSF_DTI(:,233:348); % 116 T1_CSF
FA_116=dataT1_CSF_DTI(:,349:464);   % 116 DTI_FA
MD_116=dataT1_CSF_DTI(:,465:580);   % 116 DTI_MD
L1_116=dataT1_CSF_DTI(:,581:696);   % 116 DTI_L1
L2_116=dataT1_CSF_DTI(:,697:812);   % 116 DTI_L2
L3_116=dataT1_CSF_DTI(:,813:928);   % 116 DTI_L3

V1_1_116=dataT1_CSF_DTI(:,929:1044);  % 116 DTI_V1_1
V1_2_116=dataT1_CSF_DTI(:,1045:1160); % 116 DTI_V1_2
V1_3_116=dataT1_CSF_DTI(:,1161:1276); % 116 DTI_V1_3

V2_1_116=dataT1_CSF_DTI(:,1277:1392);  % 116 DTI_V2_1
V2_2_116=dataT1_CSF_DTI(:,1393:1508); % 116 DTI_V2_2
V2_3_116=dataT1_CSF_DTI(:,1509:1624); % 116 DTI_V2_3

V3_1_116=dataT1_CSF_DTI(:,1625:1740);  % 116 DTI_V3_1
V3_2_116=dataT1_CSF_DTI(:,1741:1856); % 116 DTI_V3_2
V3_3_116=dataT1_CSF_DTI(:,1857:1972); % 116 DTI_V3_3

DSSM=dataT1_CSF_DTI(:,1973:1976);   % 4 得分
label=dataT1_CSF_DTI(:,1977);        % 1 标签
SP_116=dataT1_CSF_DTI(:,1978:2093);  %SPECT hzw 2024.11.8


isNC=(label==0);            % NC=56 PD=123 SWEDD=29
isPD=(label==2);
isSWEDD=(label==1);

IndexStirng=['Index_',groupSelect{3},'_vs_',groupSelect{4}];
load (IndexStirng); % 读取交叉验证的分组标签，之前已经分组好了的标签。

D=[T1_G_116,T1_W_116,T1_C_116,FA_116,MD_116,L1_116,L2_116,L3_116,V1_1_116,V1_2_116,V1_3_116,V2_1_116,V2_2_116,V2_3_116,V3_1_116,V3_2_116,V3_3_116,DSSM,label,SP_116];	
COMPOSE1=D(eval(groupSelect{1}),:);                 % NC=56 AD=123 SWEDD=29
COMPOSE2=D(eval(groupSelect{2}),:);                 % NC=56 AD=123 SWEDD=29      

data=[COMPOSE1;COMPOSE2];

data_T1_G_116=data(:,1:116);   % 116 灰质
data_T1_W_116=data(:,117:232); % 116 白质
data_T1_C_116=data(:,233:348); % 116 T1_CSF
data_FA_116=data(:,349:464);   % 116 DTI_FA
data_MD_116=data(:,465:580);   % 116 DTI_MD
data_L1_116=data(:,581:696);   % 116 DTI_L1
data_L2_116=data(:,697:812);   % 116 DTI_L2
data_L3_116=data(:,813:928);   % 116 DTI_L3

data_V1_1_116=data(:,929:1044);  % 116 DTI_V1_1
data_V1_2_116=data(:,1045:1160); % 116 DTI_V1_2
data_V1_3_116=data(:,1161:1276); % 116 DTI_V1_3

data_V2_1_116=data(:,1277:1392);  % 116 DTI_V2_1
data_V2_2_116=data(:,1393:1508); % 116 DTI_V2_2
data_V2_3_116=data(:,1509:1624); % 116 DTI_V2_3

data_V3_1_116=data(:,1625:1740);  % 116 DTI_V3_1
data_V3_2_116=data(:,1741:1856); % 116 DTI_V3_2
data_V3_3_116=data(:,1857:1972); % 116 DTI_V3_3

% 回归变量YSocre
depScores=data(:,1973);
sleepScores=data(:,1974);
smellScores=data(:,1975);
MoCAScores=data(:,1976);
data_label=data(:,1977);
data_SP_116=data(:,1978:2093); 


% save('data_label','data_label')

flagCOMPOSE1=ones(size(COMPOSE1,1),1);      % COMPOSE1 的标签统一为 +1
flagCOMPOSE2=-1*ones(size(COMPOSE2,1),1);	% COMPOSE2 的标签统一为 -1  （因为AD=-1 SWEDD=2 ，统一为-1）
mergeflag=[flagCOMPOSE1;flagCOMPOSE2];      % 实际与前面的data_label相等
% save('mergeflag','mergeflag')

 
% data_YScore=[data_depScores,data_sleepScores,data_smellScores,data_MoCASores,data_label];         % sleepScores,smellScores label
% save('data_YScore1','data_YScore')
data_YScore=[];
for i=1:length(regScoreSelect)
    % depScores sleepScores smellScores MoCAScores
    if(isequal(regScoreSelect{i},'depScores'))
        data_YScore=[data_YScore,eval(regScoreSelect{i})];    
    elseif(isequal(regScoreSelect{i},'sleepScores'))
        data_YScore=[data_YScore,eval(regScoreSelect{i})]; 
    elseif(isequal(regScoreSelect{i},'smellScores'))
        data_YScore=[data_YScore,eval(regScoreSelect{i})]; 
    elseif(isequal(regScoreSelect{i},'MoCAScores'))
        data_YScore=[data_YScore,eval(regScoreSelect{i})]; 
    end
end

% 无论选择谁作为回归变量，标签必须加上.
data_YScore=[data_YScore,mergeflag];
 
% % 测试代码
% test=[depScores,sleepScores,smellScores,MoCAScores];         % sleepScores,smellScores label
% save('test1','test')

X1=data_T1_G_116;   
% X1=my_convert2Sparse_impro(X1); % 每个值减均值，除以标准差.并且转换成sparse矩阵

X2=data_T1_W_116;  
% X2=my_convert2Sparse_impro(X2); % 每个值减均值，除以标准差.并且转换成sparse矩阵

M=data_T1_C_116;  
% M=my_convert2Sparse_impro(M); % 每个值减均值，除以标准差.并且转换成sparse矩阵

Y=data_FA_116;
% Y=my_convert2Sparse_impro(Y); % 每个值减均值，除以标准差.并且转换成sparse矩阵

Z=data_MD_116;
% Z=my_convert2Sparse_impro(Z); % 每个值减均值，除以标准差.并且转换成sparse矩阵

L1=data_L1_116;
% L1=my_convert2Sparse_impro(L1); % 每个值减均值，除以标准差.并且转换成sparse矩阵

L2=data_L2_116;
% L2=my_convert2Sparse_impro(L2); % 每个值减均值，除以标准差.并且转换成sparse矩阵

L3=data_L3_116;
% L3=my_convert2Sparse_impro(L3); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V1_1=data_V1_1_116;
% V1_1=my_convert2Sparse_impro(V1_1); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V1_2=data_V1_2_116;
% V1_2=my_convert2Sparse_impro(V1_2); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V1_3=data_V1_3_116;
% V1_3=my_convert2Sparse_impro(V1_3); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V2_1=data_V2_1_116;
% V2_1=my_convert2Sparse_impro(V2_1); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V2_2=data_V2_2_116;
% V2_2=my_convert2Sparse_impro(V2_2); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V2_3=data_V2_3_116;
% V2_3=my_convert2Sparse_impro(V2_3); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V3_1=data_V3_1_116;
% V3_1=my_convert2Sparse_impro(V3_1); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V3_2=data_V3_2_116;
% V3_2=my_convert2Sparse_impro(V3_2); % 每个值减均值，除以标准差.并且转换成sparse矩阵

V3_3=data_V3_3_116;
% V3_3=my_convert2Sparse_impro(V3_3); % 每个值减均值，除以标准差.并且转换成sparse矩阵

SP=data_SP_116;

%新增附加列
A=[depScores,sleepScores,smellScores,MoCAScores];
% A=my_convert2Sparse_impro(A);


% 回归变量
L=data_YScore;
% 是否将回归变量（用于特征筛选）也中心化，isYscoresCentering=1（中心化），其他不中心化
if(isYscoresCentering==1)  
%     L=my_convert2Sparse_impro(L); % 每个值减均值，除以标准差.并且转换成sparse矩阵
end


% mergeDataSlect={'X1','Z'};
mergeData=[];
for mLen=1:length(mergeDataSelect)   % mergeDataSlect 是选取的特征成分
    mergeData=[mergeData,eval(mergeDataSelect{mLen})];
end

% mergeData=[X1,X2,Y,Z,M];     % X1：灰质   X2：白质   %Y：DTI    %Z生物CSF    %M:T1CSF
mergeYScore=L;              % L：临床得分


% kFold; % 对样本进行 k 次划分
pars.k1 = subFold; % 每次划分的子集个数 kFold

parc = svmParc; % libsvm的参数设置：2的-5次方—>2的5次方
parg = svmParg;

pars.lambda1 = parsLambda1;
pars.lambda2 = parsLambda2;
pars.lambda3 = parsLambda3;
pars.lambda4 = parsLambda4;

pars.Ite=parsIte;


posSampleNum=length(flagCOMPOSE1);

% %去掉0列
% Is_zero_feature=all(mergeData==0); %hzw 2018.10.31修改
% mergeData(:,Is_zero_feature)=[];

Xpar1=mergeData(1:posSampleNum,:);      % 正样本
Xpar2=mergeData(posSampleNum+1:end,:);	% 负样本

Ypar1=mergeYScore(1:posSampleNum,:);        % 正样本 回归量
Ypar2=mergeYScore(posSampleNum+1:end,:);	% 负样本 回归量

 % depScores sleepScores smellScores MoCAScores
scoresApar1=depScores(1:posSampleNum,:);
scoresApar2=depScores(posSampleNum+1:end,:);
scoresBpar1=sleepScores(1:posSampleNum,:);
scoresBpar2=sleepScores(posSampleNum+1:end,:);
scoresCpar1=smellScores(1:posSampleNum,:);
scoresCpar2=smellScores(posSampleNum+1:end,:);
scoresDpar1=MoCAScores(1:posSampleNum,:);
scoresDpar2=MoCAScores(posSampleNum+1:end,:);

labelPar1=flagCOMPOSE1;           % 正样本 标签：类似全局变量的定义,为了并行的时候提速.
labelPar2=flagCOMPOSE2;           % 负样本 标签： 

tic
len4=length(pars.lambda4);
len3=length(pars.lambda3);  % 类似全局变量的定义,为了并行的时候提速.
len2=length(pars.lambda2);
len1=length(pars.lambda1);
TOTAL=len1*len2*len3*len4;   % 总共需要循环的次数

SelectFeaIdx=cell(kFold,pars.k1);   % SelectFeaIdx 是10次10划分交叉验证所选择的脑区下标.10*10维度
W=cell(kFold,pars.k1);              % W 是10次10划分交叉验证循环中的回归系数.10*10维度
Ite=pars.Ite;                       % 注意,这个是迭代次数.也是为了提速.

Res1=my_initialRes_impro_uar; % 初始化 hzw 2021.6.23

meanDegree=meanSelect;  % 特征选取的时候，设置的均值过滤参数


for l4=1:len4
    lamb4=pars.lambda4(l4); 
    for l3 = 1:len3 % pars.lambda3 = 1
        lamb3=pars.lambda3(l3); 
        for l2 = 1:len2 % pars.lambda2 = 1
            lamb2=pars.lambda2(l2); 
            for l1 = 1:len1 % pars.lambda1 = 1  
                lamb1=pars.lambda1(l1); % 其他两个lambda在循环开始就定义了,也是为了提速.

                singleStartTime=toc;  % 记录每次循环的开始时间(最底层的循环)            
                hasFinished=(l4-1)*len3*len2*len1+(l3-1)*len2*len1+(l2-1)*len1+l1; % 统计三层for循环完成个数
                fprintf('Doing：l4=%d/%d l3=%d/%d l2=%d/%d l1=%d/%d.\nAfter this will finish:%.2f%%(%d/%d) %.2f%%\n',l4,len4,l3,len3,l2,len2,l1,len1,hasFinished/TOTAL*100,hasFinished,TOTAL,Res1.svmUar*100);

                for kk = 1:kFold % 表示：10次 10-Kfold 划分
                    indPar1=ind1(:,kk);
                    indPar2=ind2(:,kk);

                    for ii = 1:pars.k1 % pars.k1 = 10，每次划分为10组，表示10次循环的交叉验证  %par
    %                 for ii = 1:2 % 只看一次的结果
                        % 产生训练样本 测试样本
                        [trainData,trainFlag,testData,testFlag,Atrain,Atest,Btrain,Btest,Ctrain,Ctest,Dtrain,Dtest,Ytrain] = my_Gen_samplesY2_impro(...
                            Xpar1,Xpar2,...             % X 正样本/负样本
                            Ypar1,Ypar2,...             % 正样本/负样本 处理过的 【回归矩阵=scores+lable】
                            scoresApar1,scoresApar2,...	% 正样本/负样本 depScores
                            scoresBpar1,scoresBpar2,...	% 正样本/负样本 sleepScores
                            scoresCpar1,scoresCpar2,... % 正样本/负样本 smellScores
                            scoresDpar1,scoresDpar2,... % 正样本/负样本 MoCAScores
                            labelPar1,labelPar2,...     % 正样本/负样本 label
                            indPar1,indPar2,...         % 第k次的 正样本/负样本划分
                            ii);

                        % 1. generate regression coefficient
                        % 下面的这个计算 需要将原矩阵进行转置
                        warning('off');
                        
                        %hzw 2021.4.19
                        XData=[trainData;testData];
                        
                        sparse_level=floor(116*lamb4);
                        
                        %将训练数据线性连接方式转为多数组表示
                        num_M=size(XData,2)/116;
                        XData_M=[];
                        for i=1:num_M
                            XData_M=[XData_M,{XData(:,1+(i-1)*116:i*116)}];
                        end
                        
                        %多模态数据缺失引索矩阵
                        sample_lack_index=ones(size(XData_M{1},1),num_M);
                        %样本置0比例
                        rate_set=0.15;
                        if rate_set~=0
                            rate_zero=floor(size(sample_lack_index,1)*rate_set);
                            step=size(sample_lack_index,1)/rate_zero;
                            %设置指定模态的部分数据特征为0
                            index_id=round(1:step:(size(sample_lack_index,1)-1));
                            sample_lack_index(index_id,1)=0;
                            sample_lack_index(index_id+1,2:3)=0;
%                             sample_lack_index(index_id+1,2)=0; 
                        end
                        sample_lack_index(:,num_M)=~all(XData_M{num_M}==0,2);
                        %将缺失的模态数据置0
                        for i=1:num_M
                            XData_M{i}(sample_lack_index(:,i)==0,:)=0;
                            XData_M{i}=full(my_convert2Sparse_impro2(XData_M{i}));
                        end
                        
%                         [W{kk,ii},G] = GASL_impro(XData_M,sample_lack_index,lamb1,70,lamb2,lamb3,sparse_level);

                        newTrainData=[];
                        newTestData=[];
                        for i=1:num_M
                            newTrainData = [newTrainData,XData_M{i}(1:size(trainData,1),:)];
                            newTestData = [newTestData,XData_M{i}(size(trainData,1)+1:end,:)];
                        end
                        
                        %训练数据打乱
                        random_indices_train = randperm(size(newTrainData,1));
                        newTrainData=newTrainData(random_indices_train,:);
                        trainFlag=trainFlag(random_indices_train);
                        Atrain=Atrain(random_indices_train);
                        Btrain=Btrain(random_indices_train);
                        Ctrain=Ctrain(random_indices_train);
                        Dtrain=Dtrain(random_indices_train);
                        
                        %测试数据打乱
                        random_indices_test = randperm(size(newTestData,1));
                        newTestData=newTestData(random_indices_test,:);
                        testFlag=testFlag(random_indices_test);
                        Atest=Atest(random_indices_test);
                        Btest=Btest(random_indices_test);
                        Ctest=Ctest(random_indices_test);
                        Dtest=Dtest(random_indices_test);
                        
                        %存储临时数据
                        save_path='D:\Zhongwei_Huang\My_paper\Journal_6\Experiments\first_revise\ALL_method_v5\2.GCN\dGLCN-main\dGLCN-main\codes\data\';
                        features=[newTrainData;newTestData];
                        sampleLabel=[trainFlag;testFlag];
                        sampleLabel(sampleLabel==-1)=0;
                        train_mask=[true(size(newTrainData,1),1);false(size(newTestData,1),1)];
                        val_mask=~train_mask;
                        test_mask=~train_mask;
                        
                        save([save_path,'PPMI_2classes_',num2str(ii)],'features','sampleLabel','train_mask','val_mask','test_mask');

                        
                        %python脚本调用，跨平台编程 hzw 2024.4.14
                        system(['powershell.exe -File D:\Zhongwei_Huang\My_paper\Journal_6\Experiments\first_revise\ALL_method_v5\2.GCN\run_python_script.ps1 -Index ',num2str(ii)]);
                        
%                         [W{kk,ii},G] = GASL_impro(XData_M,sample_lack_index,lamb1,70,lamb2,lamb3,sparse_level);
                        pause(5);
                        load(['D:\Zhongwei_Huang\My_paper\Journal_6\Experiments\first_revise\ALL_method_v5\2.GCN\dGLCN-main\dGLCN-main\codes\result\PPMI_2classes_result_',num2str(ii),'.mat']);
                        
                        %将权重与脑区对应起来
%                         W{kk,ii}= Recover_weight(W{kk,ii},Is_zero_feature);
                        
                        
                        %hzw 2019.7.2修改，保持分类和回归性能同步变化
                        % 注意auc、rmse都是各自返回的是parc parg循环中的最大值，与 SVMacc、Acc、Bcc无关系.
                        fsAcc(kk,ii)=test_accs;
                        fsSen(kk,ii)=test_sen;
                        fsSpec(kk,ii)=test_spe;
                        fsPrec(kk,ii)=test_prec;
                        maxFscore(kk,ii)=test_f1;
                        fsAuc1(kk,ii)=test_auc;
                        Rs1{kk,ii}.maxSvmFs_Ldec=maxAuc_Ldec;
                        Rs1{kk,ii}.maxSvmFs_Ltest=maxAuc_Ltest;
                       

                    end
                end            

                singleEndTime=toc;    % 计算每次循环所用的时间(最底层)
                fprintf('本次循环所用的时间: %f\n',singleEndTime-singleStartTime);
    %             记录每一次交叉验证的平均值              
                Res1=my_updateRes_impro_lambda_4_impro_uar_fl(Res1,fsAcc,fsSen,fsSpec,fsPrec,maxFscore,fsAuc1,...
                    Rs1,l1,l2,l3,l4,SelectFeaIdx,W);

                

            end
        end
    end    
end



toc
endTime=toc;
fprintf('%fs\n',endTime);
fprintf('%s_vs_%s\n',groupSelect{3},groupSelect{4});

save(matName,'Res1');


%    SVM for classification on class label
% function [a,b,c,d,e,f,g,h,i,j,k,AUC,R] = Com_SVM_SVR_new(NewXTrain,NewXTest,Atrain,Atest,Mtrain,Mtest,Ltrain,Ltest)
% cc:最大准确率(平方相关系数)
% rmse:最小均方误差
% SVM.acc:  accuracy 准确度
% SVM.sen:  sensitivity 敏感性——检测到正样本的数目占正标签数目的比例，有点类似R（召回率）
% SVM.spec: specificity 特异性——检测到负样本的数目占负标签数目的比例，视为 -R
% SVM.Fscore: fscore f-measure: 2PR/(P+R)
% SVM.Prec:	precision 精度 
% SVM.auc:	AUC最大值

end



