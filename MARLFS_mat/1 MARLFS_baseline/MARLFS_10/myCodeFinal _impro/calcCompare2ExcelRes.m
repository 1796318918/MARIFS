
% Excel格式要求如下：

% Feature	ACC     SEN	S   PEC     PREC	Fscore	AUC     scoreAcc	ARMSE	scoreBcc	BRMSE
% 1.1.Mat	0.737 	0.363 	0.909 	0.622 	0.440 	0.780 	0.411       3.385 	0.466       8.081 
% 1.2.Mat_	0.743 	0.357 	0.921 	0.783 	0.443 	0.717 	0.395       3.350 	0.451       8.012 

% 比较长度与最短(行数最少)的为准

xlsFile1='C:\PRO_1Cross_Res1.xlsx';
xlsFile2='C:\Res1.xlsx';
saveFile='C:\compare1.xlsx';

[~,~,xlsContent1]=xlsread(xlsFile1);
[~,~,xlsContent2]=xlsread(xlsFile2);

% 首先判断行数是否一致
if(length(xlsContent1)~=length(xlsContent2))
    fprintf('%s\n','注意：Excel行数不一致!');
end


title=xlsContent1(1,:);

xlsContent1=xlsContent1(2:end,:);
xlsContent2=xlsContent2(2:end,:);

lenRow1=size(xlsContent1,1);
lenRow2=size(xlsContent2,1);

% 行数不一致，就取最小的行
if(lenRow1<lenRow2)
    lenRow=lenRow1;
else
    lenRow=lenRow2;
end

% 列数肯定是一致的
lenColumn=size(xlsContent1,2);

k=1;
compareCell=cell(3*lenRow,lenColumn);
for i=1:lenRow
    compareCell(k,:)=xlsContent1(i,:);
    compareCell(k+1,:)=xlsContent2(i,:);
    k=k+3; % 每两行空一行
end

compareCell=[title;compareCell];

xlswrite(saveFile,compareCell); % 将need_xml_info直接转换成xls格式

