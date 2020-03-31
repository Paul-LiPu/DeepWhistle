clear all; close all;
fontlabs = 'Times New Roman';
mytitle = '';
fontsize = 15;
fontname = 'Times New Roman';

x = zeros(1,5);
for i = 1 : 5
    x(i) = 2 ^ (i - 5);
end

type = 2; % 1 for max, 2 for average

figure(1);
subplot(1, 3, 1);

if type == 1
    % % max Precision.
    y= [
        189
        186.6
        191.9
        191.4
        191.4
        ];
else
    % % average Precision.
    y= [
        185.325
        182.55
        188.225
        190.1
        191.4
        ];
end
y = y ./ 2;

plot(x, y, 'b-o', 'linewidth', 1.5, 'MarkerFaceColor','b');
title('Precision','FontSize',fontsize,'FontName', ...
    'Times New Roman','interpreter','latex');
xlabel('data amount','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex');
ylabel('Precision','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex');

subplot(1, 3, 2);
if type == 1
    % % max Recall.
    y= [
        163.7
        162.5
        162.5
        165.1
        167.3
        ];
else
    % average Recall.
    y= [
        157.175
        157.9
        161.4
        160.95
        167.3
        ];
end
y = y ./ 2;

plot(x, y, 'b-o', 'linewidth', 1.5, 'MarkerFaceColor','b');
title('Recall','FontSize',fontsize,'FontName', ...
    'Times New Roman','interpreter','latex');
xlabel('data amount','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex');
ylabel('Recall','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex');


subplot(1, 3, 3);
if type == 1
    % max F-score.
    y= [
        171.73
        171.75
        175.79
        176.10
        178.33
        ];
else
    % % average F-score.
    y= [
        169.87
        169.24
        173.67
        174.14
        178.33
        ];
end
y = y ./ 2;

plot(x, y, 'b-o', 'linewidth', 1.5, 'MarkerFaceColor','b');
title('F-measure','FontSize',fontsize,'FontName', ...
    'Times New Roman','interpreter','latex');
xlabel('data amount','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex');
ylabel('F-measure','FontSize',fontsize,'FontName',fontlabs, ...
    'interpreter','latex');