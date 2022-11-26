%% creditSVM.m
% ps command:
% matlab -nosplash -nodesktop -sd d:\Dev\ML-AI\CreditScoreML -r "run('.\creditSVM.m');"
% https://www.mathworks.com/help/stats/classification-with-imbalanced-data.html
function creditSVM
clear all, close all hidden

% Load the csv file
cleanCred = readtable('clean_credit_score_classification.csv');
%trainCred = readtable('train.csv');
%testCred = readtable('test.csv');

% Run zscores on numerical values:
trainCredReg = cleanCred;
% zscores on numerical values: age[5],moInhandSal[9],numBankAcct[10],numCredCards[11],
%                              numDelayPay[16],numOutstandDebt[20],credHistAge[22]
trainCredReg(:, [5 8 9 10 11 16 20 22]) = ...
    array2table(zscore(table2array(cleanCred(:, [5 8 9 10 11 16 20 22]))));
% cat vars standardization: (convert)annualIncome[8],payMinAmount[23]


% Put Data in X and Y
xCredTrain = trainCredReg(:, [5 8 9 10 11 16 20 22]);
yCred = trainCredReg(:, end);
tabulate(yCred.Credit_Score)

%% Build Model
%RBF_SVM = fitcecoc(xCredTrain, yCredTrain);
rng(10)         % For reproducibility
part = cvpartition(yCred.Credit_Score,'Holdout',0.50);
istrain = training(part); % Data for fitting
istest = test(part);      % Data for quality assessment
N = sum(istrain);         % Number of observations in the training sample
t = templateTree('MaxNumSplits',N);
rusTree = fitcensemble(xCredTrain(istrain,:),yCred.Credit_Score(istrain),'Method','RUSBoost', ...
    'NumLearningCycles',1000,'Learners',t,'LearnRate',0.1,'nprint',100);

figure;
plot(loss(rusTree,xCredTrain(istest,:),yCred.Credit_Score(istest),'mode','cumulative'));

grid on;
xlabel('Number of trees');
ylabel('Test classification error');

Yfit = predict(rusTree,xCredTrain(istest,:));

confusionchart(yCred.Credit_Score(istest),Yfit,'Normalization','row-normalized','RowSummary','row-normalized')

pause = "place break on this line"
