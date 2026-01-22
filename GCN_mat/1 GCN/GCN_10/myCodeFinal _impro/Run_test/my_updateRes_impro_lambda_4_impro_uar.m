function Res=my_updateRes_impro_lambda_4_impro_uar(Res,Acc,Sen,Spec,Prec,Fscore,Auc,...
    maxAcc,minArmse,maxBcc,minBrmse,maxCcc,minCrmse,maxDcc,minDrmse,Rs,Ra,Rb,Rc,Rd,...
    l1,l2,l3,l4,SelectFeaIdx,W)

% 2016.04.07新增了一项W

% 注意这里的{Acc,Sen,Spec,Prec,Fscore}是一个集合，要么对应的都是maxAcc，要么都是maxFscore
% 而auc是单独的，它是每次parc parg循环中，取的最大值。

    %----------------------- SVM --------------------------
    meanOfAcc = mean(Acc(:));
    meanOfSen = mean(Sen(:));
    meanOfSpec = mean(Spec(:));
    meanOfPrec= mean(Prec(:)); 
    meanOfFscore = mean(Fscore(:));
    meanOfAuc = mean(Auc(:));
    
    stdOfAcc = std(Acc(:));
    stdOfSen = std(Sen(:));
    stdOfSpec = std(Spec(:));
    stdOfPrec = std(Prec(:));
    stdOfFscore = std(Fscore(:));
    stdOfAuc = std(Auc(:));
    
    %使用Uar更新 hzw 2021.6.17
    meanOfUar = mean((Sen+Spec)/2);
    stdOfUar = std((Sen+Spec)/2);
    
    % update SVM 
    %hzw 2019.7.2 修订，保持Fscore最大时，更新所有值
    if Res.svmUar < meanOfUar
        Res.svmAcc = meanOfAcc;
        Res.svmAcc_Std = stdOfAcc;    %hzw 2019.7.2添加
        Res.svmSen = meanOfSen;
        Res.svmSen_Std = stdOfSen;     %hzw 2019.7.2添加
        Res.svmSpec = meanOfSpec;
        Res.svmSpec_Std = stdOfSpec;   %hzw 2019.7.2添加
        Res.svmPrec = meanOfPrec;
        Res.svmPrec_Std = stdOfPrec;   %hzw 2019.7.2添加
        %添加Uar hzw 2021.6.17
        Res.svmUar=meanOfUar;
        Res.svmUar_Std=stdOfUar;
        
        Res.svmFscore = meanOfFscore;
        Res.svmFscore_Std = stdOfFscore;
        Res.svmAuc = meanOfAuc;
        Res.svmAuc_Std = stdOfAuc;   %hzw 2019.7.2添加
        Res.svm_Fea = SelectFeaIdx;
        Res.svm_W = W;               
        Res.svm_Rs=Rs;
        Res.svmFscore_Lamb=[l1,l2,l3,l4];   %hzw 2019.7.2添加
    end
    
    
    %---------------------- scoreA ------------------------
    meanOfmaxAcc = mean(maxAcc(:));
    stdOfmaxAcc = std(maxAcc(:)); 
    meanOfminArmse = mean(minArmse(:));   %hzw 2019.7.2
    stdOfminArmse = std(minArmse(:));
    if Res.maxAcc <= meanOfmaxAcc && meanOfmaxAcc <= 1      %hzw 2019.7.3           
        Res.maxAcc = meanOfmaxAcc;
        Res.maxAcc_Std = stdOfmaxAcc;
        Res.minArmse = meanOfminArmse;   %hzw 2019.7.2
        Res.minArmse_Std = stdOfminArmse;
        Res.maxAcc_Ra=Ra;
        Res.maxAcc_Lamb=[l1,l2,l3,l4];
    end

    
    %---------------------- scoreB ------------------------
    meanOfmaxBcc = mean(maxBcc(:));
    stdOfmaxBcc = std(maxBcc(:));
    meanOfminBrmse = mean(minBrmse(:));   %hzw 2019.7.2
    stdOfminBrmse = std(minBrmse(:));
    if Res.maxBcc <= meanOfmaxBcc && meanOfmaxBcc <= 1    %hzw 2019.7.3            
        Res.maxBcc = meanOfmaxBcc;
        Res.maxBcc_Std = stdOfmaxBcc;
        Res.minBrmse = meanOfminBrmse;
        Res.minBrmse_Std = stdOfminBrmse;
        Res.maxBcc_Rb=Rb;
        Res.maxBcc_Lamb=[l1,l2,l3,l4];
    end
    
    
    %---------------------- scoreC ------------------------
    meanOfmaxCcc = mean(maxCcc(:));
    stdOfmaxCcc = std(maxCcc(:));  
    meanOfminCrmse = mean(minCrmse(:));   %hzw 2019.7.2
    stdOfminCrmse = std(minCrmse(:));
    if Res.maxCcc <= meanOfmaxCcc && meanOfmaxCcc <= 1    %hzw 2019.7.3         
        Res.maxCcc = meanOfmaxCcc;
        Res.maxCcc_Std = stdOfmaxCcc;
        Res.minCrmse = meanOfminCrmse;
        Res.minCrmse_Std = stdOfminCrmse;
        Res.maxCcc_Rc=Rc;
        Res.maxCcc_Lamb=[l1,l2,l3,l4];
    end
    
    
    %---------------------- scoreD ------------------------
    meanOfmaxDcc = mean(maxDcc(:));
    stdOfmaxDcc = std(maxDcc(:));  
    meanOfminDrmse = mean(minDrmse(:));
    stdOfminDrmse = std(minDrmse(:));
    if Res.maxDcc <= meanOfmaxDcc && meanOfmaxDcc <= 1    %hzw 2019.7.3            
        Res.maxDcc = meanOfmaxDcc;
        Res.maxDcc_Std = stdOfmaxDcc;
        Res.minDrmse = meanOfminDrmse;
        Res.minDrmse_Std = stdOfminDrmse;
        Res.maxDcc_Rd=Rd;
        Res.maxDcc_Lamb=[l1,l2,l3,l4];
    end
    
    
    
end



















