function Res=my_updateRes_impro_lambda_4_impro_uar_fl(Res,Acc,Sen,Spec,Prec,Fscore,Auc,...
    Rs,l1,l2,l3,l4,SelectFeaIdx,W)

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
    
    
   
    
    
    
end



















