clear all; close all;
fontlabs = 'Times New Roman';  
mytitle = '';
fontsize = 15;
fontname = 'Times New Roman';

x = zeros(1,5);
for i = 1 : 5
    x(i) = 2 ^ (i - 5);
end

y= [178.33 
176.10
175.79 
171.75 
171.73];
y = y(end:-1:1);
y = y ./ 2;


figure(1);
plot(x, y, 'b-o', 'linewidth', 1.5, 'MarkerFaceColor','b');
title('Average F-measure','FontSize',fontsize,'FontName', ...
    'Times New Roman','interpreter','latex'); 
xlabel('data amount','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex'); 
ylabel('F-measure','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex'); 
% set(gca, 'XDir','reverse')