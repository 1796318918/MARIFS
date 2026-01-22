
% data		num		label
% NC 		65		  0
% PD 		156		  2
% SWEDD 	35	      1
% 
% 
% ºÏ¼Æ		256

% % PDvsNC
% ind1 = crossvalind('Kfold', 156, 10);
% ind2 = crossvalind('Kfold', 65, 10);
% save('Index_PD_vs_NC','ind1','ind2');


% % SWEDDvsNC
% ind1 = crossvalind('Kfold', 35, 10);
% ind2 = crossvalind('Kfold', 65, 10);
% save('Index_SWEDD_vs_NC','ind1','ind2');


% % PDvsSWEDD
ind1 = crossvalind('Kfold', 156, 10);
ind2 = crossvalind('Kfold', 35, 10);
save('Index_PD_vs_SWEDD','ind1','ind2');
