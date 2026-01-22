function original_W= Recover_weight(current_W,Is_zero_feature)

[feature_num,task_length]=size(current_W);    %得到目前权值行列维度
original_feature_num=numel(Is_zero_feature);  %得到原始权重维度
i=1;
j=1;
original_W=[];
append_zero=zeros(1,task_length);
while i<= original_feature_num
    if Is_zero_feature(1,i)==0
        original_W=[original_W;current_W(j,:)];
        i=i+1;
        j=j+1;
    else
        original_W=[original_W;append_zero];
        i=i+1;
    end
end

if feature_num ~= j-1
    display('Error!!!');
end