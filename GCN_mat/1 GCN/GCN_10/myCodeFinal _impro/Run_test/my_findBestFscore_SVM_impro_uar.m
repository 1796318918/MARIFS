function [fsAcc,fsSen,fsSpec,fsPrec,fsAuc,maxFscore,fsLdec,fsLtest,fsTP,fsFP,fsBest_X,fsBest_Y]=...
    my_findBestFscore_SVM_impro_uar(matAcc,matSen,matSpec,matPrec,matAuc,matFscore,matLdec,matLtest,matTP,matFP)

    
%     % -------------------- Fscore -----------------------------
%     maxFscore=max(matFscore(:)); % 找出最大fscores.可能有重复的
%     seqIndex=find(matFscore==maxFscore); % 找出所有重复的最大值的顺序索引(按列)
%     [x,y]=ind2sub(size(matFscore),seqIndex); % 将索引转换成矩阵的坐标,注意 x,y是向量
    
    %使用UAR更新 hzw 2021.6.17
    matUar=(matSen+matSpec)/2;
    maxUar=max(matUar(:));
    seqIndex=find(matUar==maxUar);
    [x,y]=ind2sub(size(matUar),seqIndex);

    % 既然fscores最大值，存在重复，那么我们再看  acc 准确率 指标
    maxFscore=[];
    fsAcc=[];
    fsSen=[];
    fsSpec=[];
    fsPrec=[];   
    fsAuc=[];
    fsLdec={};
    fsLtest={};
    fsTP={};
    fsFP={};
    
    fsBest_X=[];   % 记录最大值X的坐标
    fsBest_Y=[];   % 记录最大值Y的坐标
    for i=1:length(x)  % 最大的fscores，根据其下标找出所有对应的其他指标
        
        
        fsBest_X=[fsBest_X,x(i)];
        fsBest_Y=[fsBest_Y,y(i)];
        
        maxFscore=[maxFscore,matFscore(x(i),y(i))];
        fsAcc=[fsAcc,matAcc(x(i),y(i))];        
        fsSen=[fsSen,matSen(x(i),y(i))];
        fsSpec=[fsSpec,matSpec(x(i),y(i))];
        fsPrec=[fsPrec,matPrec(x(i),y(i))];        
        fsAuc=[fsAuc,matAuc(x(i),y(i))];
        
        fsLdec=[fsLdec,matLdec{x(i),y(i)}];
        fsLtest=[fsLtest,matLtest{x(i),y(i)}];
        fsTP=[fsTP,matTP{x(i),y(i)}];
        fsFP=[fsFP,matFP{x(i),y(i)}];
    end
    
    [~,maxMapIndex]=max(fsAcc(:)); % 按出最大 acc .可能有重复的.如果有重复的就选择第一个，不再进一步看其他指标了
    maxFscore=maxFscore(maxMapIndex);
    fsAcc=fsAcc(maxMapIndex);
    fsSen=fsSen(maxMapIndex);
    fsSpec=fsSpec(maxMapIndex);
    fsPrec=fsPrec(maxMapIndex);   
    fsAuc=fsAuc(maxMapIndex);
    
    fsLdec=fsLdec{maxMapIndex};
    fsLtest=fsLtest{maxMapIndex};
    fsTP=fsTP{maxMapIndex};
    fsFP=fsFP{maxMapIndex};
    
    fsBest_X=fsBest_X(maxMapIndex);
    fsBest_Y=fsBest_Y(maxMapIndex);

end









