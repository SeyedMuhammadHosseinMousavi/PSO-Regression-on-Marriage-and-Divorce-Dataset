%% PSO Regression on Marriage and Divorce Dataset
% This data contains 31 columns (100x31). The first 30 columns are features (Inputs)
% and last one is Targets.
% Relevant Dataset Paper or Citation Request:
% Mousavi, S. M. H., MiriNezhad, S. Y., & Lyashenko, V, An evolutionary-based
% adaptive Neuro-fuzzy expert system as a family counselor before marriage with 
% the aim of divorce rate reduction, 2nd International Conference on Research
% Knowledge Base in Computer, Tehran, Iran, (2017).
%-----------------------------------------------------------------------
clear;
Data=load('Marriage_Divorce_DB.mat');
Inputs=Data.Marriage_Divorce_DB(:,1:end-1);
Targets=Data.Marriage_Divorce_DB(:,end);
%% Learning 
n = 9; % Neurons
%----------------------------------------
% 'trainlm'	    Levenberg-Marquardt
% 'trainbr' 	Bayesian Regularization (good)
% 'trainrp'  	Resilient Backpropagation
% 'traincgf'	Fletcher-Powell Conjugate Gradient
% 'trainoss'	One Step Secant (good)
% 'traingd' 	Gradient Descent
% Creating the NN ----------------------------
net = feedforwardnet(n,'trainoss');
%---------------------------------------------
% configure the neural network for this dataset
[net tr]= train(net,Inputs', Targets');
perf = perform(net,Inputs, Targets); % mse
% Current NN Weights and Bias
Weights_Bias = getwb(net);
% MSE Error for Current NN
Outputs=net(Inputs');
Outputs=Outputs';
% Final MSE Error and Correlation Coefficients (CC)
Err_MSE=mse(Targets,Outputs);
CC1= corrcoef(Targets,Outputs);
CC1= CC1(1,2);
%-----------------------------------------------------
%% Nature Inspired Regression
% Create Handle for Error
h = @(x) NMSE(x, net, Inputs', Targets');
sizenn=size(Inputs);sizenn=sizenn(1,1);
%-----------------------------------------
%% Please select BBO (bbo) or PSO (pso)
[x, cost] = pso(h, sizenn*n+n+n+1);
%-----------------------------------------
net = setwb(net, x');
% Optimized NN Weights and Bias
getwb(net);
% Error for Optimized NN
Outputs2=net(Inputs');
Outputs2=Outputs2';
% Final MSE Error and Correlation Coefficients (CC)
Err_MSE2=mse(Targets,Outputs2);
CC2= corrcoef(Targets,Outputs2);
CC2= CC2(1,2);

%% Plot Regression
figure('units','normalized','outerposition',[0 0 1 1])
% Normal
subplot(1,2,1)
[population2,gof] = fit(Targets,Outputs,'poly3');
plot(Targets,Outputs,'o',...
'LineWidth',1,...
'MarkerSize',8,...
'MarkerEdgeColor','r',...
'MarkerFaceColor',[0.3,0.9,0.1]);
title(['Normal R =  ' num2str(1-gof.rmse)],['Normal MSE =  ' num2str(Err_MSE)]);
hold on
plot(population2,'b-','predobs');
xlabel('Targets');ylabel('Outputs');   grid on;
ax = gca; 
ax.FontSize = 12; ax.LineWidth=2;
legend({'Normal Regression'},'FontSize',12,'TextColor','blue');hold off
% PSO
subplot(1,2,2)
[population3,gof3] = fit(Targets,Outputs2,'poly3');
plot(Targets,Outputs2,'o',...
'LineWidth',1,...
'MarkerSize',8,...
'MarkerEdgeColor','g',...
'MarkerFaceColor',[0.9,0.3,0.1]);
title(['PSO R =  ' num2str(1-gof3.rmse)],['PSO MSE =  ' num2str(Err_MSE2)]); 
hold on
plot(population3,'b-','predobs');
xlabel('Targets');ylabel('Outputs');   grid on;
ax = gca; 
ax.FontSize = 12; ax.LineWidth=2;
legend({'PSO Regression'},'FontSize',12,'TextColor','blue');hold off

% Correlation Coefficients
fprintf('Normal Correlation Coefficients Is =  %0.4f.\n',CC1);
fprintf('PSO Correlation Coefficients Is =  %0.4f.\n',CC2);
