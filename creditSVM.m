%% creditSVM.m
% ps command:
% matlab -nosplash -nodesktop -sd d:\Dev\ML-AI\CreditScoreML -r "run('.\creditSVM.m');"
% https://www.mathworks.com/help/stats/classification-with-imbalanced-data.html
function a8
%clear all 
close all hidden

% Load the csv file
%% Credit Section
origCred = readtable('clean_credit_score_classification.csv');
Y = origCred(:,end);
%origCred(:,end) = []; % commented to leave in score (that we are trying find)
tabulate(Y.Credit_Score)

cleanCred = readtable('clean_credit_score_classification.csv');
trainCred = readtable('train.csv');
testCred = readtable('test.csv');

%% Create New Data
if 1
    histCred  = readtable('histCred.csv');
%if exist('histCred','var') == 1
else
    disp("Creating new dataset")
    % Create 30 features per customer (to incorporate customer history)
    % --Instead of treating each line as a single entry
    % 10 features per month, 3 months per customer
    % Loop on list, when we find a customer name, combine last three months ()
    %% numerical vars:
    % moInhandSal[9],numBankAcct[10],numCredCards[11],
    % numDelayPay[16],numOutstandDebt[20],credHistAge[22]
    %% categorical vars: 
    % (convert)annualIncome[8],payMinAmount[23]
    histCredOutput = array2table(zeros(0,1));
    histCredOutput.Properties.VariableNames = {'Score'};
    histCred = array2table(zeros(0,28));
    histCred.Properties.VariableNames = { ...
        ... One Time Variables (no need for historical records) ...
        'Occupation','Latest_Age', 'Latest_Income', ...
        ... Numerical Variables ...
        'inhand1','inhand2','inhand3', ...
        'numBank1','numBank2','numBank3', ...
        'numCard1','numCard2','numCard3', ...
        'numDelay1','numDelay2','numDelay3', ...
        'numDebt1','numDebt2','numDebt3', ...
        'credAge1','credAge2','credAge3', ...
        'annIncome1','annIncome2','annIncome3', ...
        ... Categorical Variables ...
        'payMin1','payMin2','payMin3', ...
        ... Prediction Variable (Y) ...
        'creditScore' ...
    };
    
    table_size = height(origCred);
    rows = table_size(1);
    currentUser = 'QWERTY';
    first = 1;
    userCount = 1; % start at one for indexing beginning at 1
    for row = 1:(rows)
    
        if (~ismember(origCred{row,4}, currentUser)) && (~first) % if User is different:
            % New user detected so stop grabbing data and store it
            first = 1; % reset so new user can be created
            %break; don't break because we need to hit the next if to create a
            %new user after we store this one
            userCount = userCount+1;
            histCred = [histCred;struct2table(userStruct)];
            histCredOutput = [histCredOutput;struct2table(userStructScore)];
        end
    
        if (~ismember(origCred{row,4}, currentUser)) && (first) % if User is different:
            currentUser = origCred{row,"Name"};
            first = 0; % set first to false so we can properly compile user history
            %userStruct(1,1).Name = origCred{row,"Name"};
        elseif ismember(origCred{row,4}, currentUser)
            if ismember(origCred{row,"Month"}, 8) % if Last month (8)
                userStructScore(1,1).Score = origCred{row,"Credit_Score"}; % store latest score (our prediction)
                userStruct(1,1).creditScore = origCred{row,"Credit_Score"};
                userStruct(1,1).Occupation = origCred{row,"Occupation"};
                userStruct(1,1).Latest_Age = origCred{row,"Age"}; % store latest age
                userStruct(1,1).Latest_Income = origCred{row,"Annual_Income"}; % store latest income
                userStruct(1,1).inhand3 = origCred{row,"Monthly_Inhand_Salary"};
                userStruct(1,1).numBank3 = origCred{row,"Num_Bank_Accounts"};
                userStruct(1,1).numCard3 = origCred{row,"Num_Credit_Card"};
                userStruct(1,1).numDelay3 = origCred{row,"Num_of_Delayed_Payment"};
                userStruct(1,1).numDebt3 = origCred{row,"Outstanding_Debt"};
                userStruct(1,1).credAge3 = origCred{row,"Credit_History_Age"};
                userStruct(1,1).annIncome3 = origCred{row,"Annual_Income"};
                userStruct(1,1).payMin3 = origCred{row,"Payment_of_Min_Amount"};
                % Add amount invested monthly [25]
                % Add payment behavior [26]
                % Add credit mix [19]
            elseif ismember(origCred{row,"Month"}, 7) % if Last month (7)
                userStruct(1,1).inhand2 = origCred{row,"Monthly_Inhand_Salary"};
                userStruct(1,1).numBank2 = origCred{row,"Num_Bank_Accounts"};
                userStruct(1,1).numCard2 = origCred{row,"Num_Credit_Card"};
                userStruct(1,1).numDelay2 = origCred{row,"Num_of_Delayed_Payment"};
                userStruct(1,1).numDebt2 = origCred{row,"Outstanding_Debt"};
                userStruct(1,1).credAge2 = origCred{row,"Credit_History_Age"};
                userStruct(1,1).annIncome2 = origCred{row,"Annual_Income"};
                userStruct(1,1).payMin2 = origCred{row,"Payment_of_Min_Amount"};
            elseif ismember(origCred{row,"Month"}, 6) % if Last month (6)
                userStruct(1,1).inhand1 = origCred{row,"Monthly_Inhand_Salary"};
                userStruct(1,1).numBank1 = origCred{row,"Num_Bank_Accounts"};
                userStruct(1,1).numCard1 = origCred{row,"Num_Credit_Card"};
                userStruct(1,1).numDelay1 = origCred{row,"Num_of_Delayed_Payment"};
                userStruct(1,1).numDebt1 = origCred{row,"Outstanding_Debt"};
                userStruct(1,1).credAge1 = origCred{row,"Credit_History_Age"};
                userStruct(1,1).annIncome1 = origCred{row,"Annual_Income"};
                userStruct(1,1).payMin1 = origCred{row,"Payment_of_Min_Amount"};
            end
        end
    end
    
    writetable(histCred,'histCred.csv','Delimiter',',','QuoteStrings',true)
end

%% Create equal training set
xEvenHistCred = array2table(zeros(0,28));
yEvenHistCred = array2table(zeros(0,1));
yEvenHistCred.Properties.VariableNames = {'Score'};
xEvenHistCred.Properties.VariableNames = { ...
    ... One Time Variables (no need for historical records) ...
    'Occupation','Latest_Age', 'Latest_Income', ...
    ... Numerical Variables ...
    'inhand1','inhand2','inhand3', ...
    'numBank1','numBank2','numBank3', ...
    'numCard1','numCard2','numCard3', ...
    'numDelay1','numDelay2','numDelay3', ...
    'numDebt1','numDebt2','numDebt3', ...
    'credAge1','credAge2','credAge3', ...
    'annIncome1','annIncome2','annIncome3', ...
    ... Categorical Variables ...
    'payMin1','payMin2','payMin3', ...
    ... Prediction Variable (Y) ...
    'creditScore' ...
};
table_size = height(histCred);
rows = table_size(1);
poorVal = 0;
stanVal = 0;
goodVal = 0;
for row = 1:(rows)
    if (ismember(histCred{row,"creditScore"}, 'Poor')) && poorVal == 0
        poorVal = 1;
        xEvenHistCred = [xEvenHistCred;histCred(row,:)];
    elseif (ismember(histCred{row,"creditScore"}, 'Standard')) && stanVal == 0
        stanVal = 1;
        xEvenHistCred = [xEvenHistCred;histCred(row,:)];
    elseif (ismember(histCred{row,"creditScore"}, 'Good')) && goodVal == 0
        goodVal = 1;
        xEvenHistCred = [xEvenHistCred;histCred(row,:)];
    end

    if (poorVal == 1) && (stanVal == 1) && (goodVal == 1)
        poorVal = 0;
        stanVal = 0;
        goodVal = 0;
    end
end
yEvenHistCred = xEvenHistCred(:,end);
tabulate(yEvenHistCred.creditScore)


yCredTrain = xEvenHistCred(:,end);
yCredTest = histCred(:,end);
% Run zscores on numerical values:
even_testCredReg = xEvenHistCred;
all_testCredReg = histCred;
even_testCredReg(:, 2:24) = array2table(zscore(table2array(even_testCredReg(:, [2:24]))));
all_testCredReg(:, 2:24) = array2table(zscore(table2array(all_testCredReg(:, [2:24]))));

% cat vars standardization: (convert)annualIncome[8],payMinAmount[23]

% Put Data in X and Y
%xCredTrain = testCredReg(:, :);
%xCredTrain = testCredReg(:, [1 2 3 6 9 12 15 18 21 24 25 26 27]); %w/ standardization
xCredTrain = even_testCredReg(:, [1 2 3 6 9 12 15 18 21 24 27]); %w/ standardization
xCredTest = all_testCredReg(:, [1 2 3 6 9 12 15 18 21 24 27]); %w/ standardization
%xCredTrain = histCred(:, [1 2 3 6 9 12 15 18 21 24 25 26 27]); %w/o standardization


% Build RBF SVM
%RBF_SVM = fitcsvm(X, Y, 'BoxConstraint', ContraintBox, 'KernelFunction', 'RBF', 'KernelScale', 1);
%mdl = fitcecoc(xCredTrain, yCredTrain);
rng(5)         % For reproducibility
part = cvpartition(yCredTrain.creditScore,'Holdout',1.00);
istrain = training(part); % Data for fitting
istest = test(part);      % Data for quality assessment
N = sum(istrain);         % Number of observations in the training sample
t = templateTree('MaxNumSplits',N);

% https://www.mathworks.com/help/stats/framework-for-ensemble-learning.html#bsw73lr
%mdl = fitcensemble(xCredTrain(istrain,:),yCredTrain.creditScore(istrain),'Method','RUSBoost','NumLearningCycles',1000,'Learners',t,'LearnRate',0.1,'nprint',250);
mdl = fitcensemble(xCredTrain(istrain,:),yCredTrain.creditScore(istrain),'Method','Subspace','NumLearningCycles',1000,'Learners','knn','nprint',250);
%mdl = fitcensemble(xCredTrain(istrain,:),yCredTrain.creditScore(istrain), 'Method', 'AdaBoostM2');
%mdl = fitcecoc(xCredTrain(istrain,:), yCredTrain.creditScore(istrain));
%mdl = fitcensemble(xCredTrain(istrain,:),yCredTrain.creditScore(istrain));
%% 15.5, 89.6, 78.7

figure;
%plot(loss(mdl,xCredTrain(istest,:),yCredTrain.creditScore(istest),'mode','cumulative')); 

grid on;
xlabel('Number of trees');
ylabel('Test classification error');

Yfit = predict(mdl,xCredTest(:,:));

confusionchart(yCredTest.creditScore(:),Yfit,'Normalization','row-normalized','RowSummary','row-normalized')
%confusionchart(yCredTest.creditScore(:),Yfit)



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


