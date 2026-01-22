function [maxCc,ccRmse,ccDec,ccTest,ccBest_X,ccBest_Y]=...
    my_findBestCc_SVR_impro2(matCc,matRmse,matDec,matTest)

    % 找出最大cc(相关系数)以及对应的rmse、Ldec、Ltest、parg parc参数下标
    % 同时找出最小的rmse（均方误差），它没有对应的值


%     % hzw 2019.7.2注释该内容，原因是CC和RMSE应该同时变化
%     minRmse=min(matRmse(:));

    [~,cc_Index]=sort(matCc(:),'descend');
    maxMapIndex=-1;
    for i=1:length(cc_Index)
        if matRmse(cc_Index(i))<=20
            maxMapIndex=cc_Index(i);
            break;
        end
    end
    if maxMapIndex==-1
        maxCc=0;
        ccRmse=8888;
        ccDec=[];
        ccTest=[];
        ccBest_X=0;
        ccBest_Y=0;
    else
        [x,y]=ind2sub(size(matCc),maxMapIndex);
        maxCc=matCc(x,y);
        ccRmse=matRmse(x,y);
        ccDec=matDec{x,y};
        ccTest=matTest{x,y};
        ccBest_X=x;
        ccBest_Y=y;
    end
end









