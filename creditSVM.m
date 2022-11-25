%% creditSVM.m
% ps command:
% matlab -nosplash -nodesktop -sd d:\Dev\ML-AI\CreditScoreML -r "run('.\creditSVM.m');"
% https://www.mathworks.com/help/stats/classification-with-imbalanced-data.html
function a8
clear all, close all hidden

% Load the csv file
%% Credit Section
origCred = readtable('clean_credit_score_classification.csv');
Y = origCred(:,end);
origCred(:,end) = [];
tabulate(Y.Credit_Score)

cleanCred = readtable('clean_credit_score_classification.csv');
trainCred = readtable('train.csv');
testCred = readtable('test.csv');

%% Credit Sections
yCredTrain = trainCred(:,[28]); %Credit Score: Good/Standard/Poor --> Categorical
%%%
yCredTrain = cleanCred(:,[28]); %Credit Score: Good/Standard/Poor --> Categorical
%%%

% Run zscores on numerical values:
%% Credit Sections 
trainCredReg = trainCred;
testCredReg = testCred;
%%%
trainCredReg = cleanCred;
testCredReg = cleanCred;
%%%
% zscores on numerical values: age[5],moInhandSal[9],numBankAcct[10],numCredCards[11],
%                              numDelayPay[16],numOutstandDebt[20],credHistAge[22]
%trainCredReg(:, [5 8 9 10 11 16 20 22]) = ...
%    array2table(zscore(table2array(trainCred(:, [5 8 9 10 11 16 20 22]))));
%testCredReg(:, [5 8 9 10 11 16 20 22]) = ...
%    array2table(zscore(table2array(testCred(:, [5 8 9 10 11 16 20 22]))));
%%%
trainCredReg(:, [5 8 9 10 11 16 20 22]) = ...
    array2table(zscore(table2array(cleanCred(:, [5 8 9 10 11 16 20 22]))));
testCredReg(:, [5 8 9 10 11 16 20 22]) = ...
    array2table(zscore(table2array(cleanCred(:, [5 8 9 10 11 16 20 22]))));
%%%
% cat vars standardization: (convert)annualIncome[8],payMinAmount[23]


% Put Data in X and Y
%% Credit Sections 
xCredTrain = trainCredReg(:, [5 8 9 10 11 16 20 22]);
%xCredTrain = table2array(xCredTrain);
xCredTest = trainCredReg(:, [5 8 9 10 11 16 20 22]);
xCredTest = table2array(xCredTest);
yCredTrain = table2array(yCredTrain);

% Setup SVM Inputs
ContraintBox = 1;

% Build RBF SVM
%RBF_SVM = fitcsvm(X, Y, 'BoxConstraint', ContraintBox, 'KernelFunction', 'RBF', 'KernelScale', 1);
%% Credit Sections 
%RBF_SVM = fitcecoc(xCredTrain, yCredTrain);
rng(10,'twister')         % For reproducibility
part = cvpartition(Y.Credit_Score,'Holdout',0.50);
istrain = training(part); % Data for fitting
istest = test(part);      % Data for quality assessment
N = sum(istrain);         % Number of observations in the training sample
t = templateTree('MaxNumSplits',N);
rusTree = fitcensemble(xCredTrain(istrain,:),Y.Credit_Score(istrain),'Method','RUSBoost', ...
    'NumLearningCycles',1000,'Learners',t,'LearnRate',0.1,'nprint',100);

figure;
plot(loss(rusTree,xCredTrain(istest,:),Y.Credit_Score(istest),'mode','cumulative'));

grid on;
xlabel('Number of trees');
ylabel('Test classification error');

Yfit = predict(rusTree,xCredTrain(istest,:));

confusionchart(Y.Credit_Score(istest),Yfit,'Normalization','row-normalized','RowSummary','row-normalized')



% Prepare Grid for Plotting
gap = 0.01;
%[x1Grid,x2Grid] = meshgrid(min(X(:,1)) : gap : max(X(:,1)), min(X(:,2)) : gap : max(X(:,2)));
%% Credit Sections
[x1Grid,x2Grid] = meshgrid(min(xCredTrain(:,1)) : gap : max(xCredTrain(:,1)), min(xCredTrain(:,2)) : gap : max(xCredTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

% Reformat X and Y
%x_cell = num2cell(X);
%y_cell = num2cell(Y);
%% Credit Sections
x_cell = num2cell(xCredTrain);
y_cell = num2cell(yCredTrain);

% Plot RBF SVM
%Plot2DDecisionBoundary(RBF_SVM, xCredTrain, yCredTrain, xGrid, 'RBF SVM')

% Get Accuracy, Precision, and Recall for Each Class
[predictClasses,proba] = predict(RBF_SVM, xCredTest);
CFM_Stats(yCredTrain, predictClasses)

% 5. Roc Curves
[xpos, ypos, T, AUC0] = perfcurve(yCredTrain, proba(:, 1), 'Good');
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05]), xlabel('\bf FP rate'),  ylabel('\bf TP rate')
title('\bf ROC for class Good')
[xpos, ypos, T, AUC1] = perfcurve(yCredTrain, proba(:, 2), 'Standard');
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05]), xlabel('\bf FP rate'),  ylabel('\bf TP rate')
title('\bf ROC for class Standard')
[xpos, ypos, T, AUC2] = perfcurve(yCredTrain, proba(:, 2), 'Poor');
figure, plot(xpos, ypos)
xlim([-0.05 1.05]), ylim([-0.05 1.05]), xlabel('\bf FP rate'),  ylabel('\bf TP rate')
title('\bf ROC for class Poor')

return;
              
function Plot2DDecisionBoundary(model, X, Y, gridIn, plotTitle)
[yh, ~] = predict(model, gridIn);
gscatter(gridIn(:,1), gridIn(:,2), yh, 'cg'), hold on,
gscatter(X(:,1), X(:,2), Y, 'rb', '.', 10);
title(['\bf' plotTitle])
axis tight, drawnow


