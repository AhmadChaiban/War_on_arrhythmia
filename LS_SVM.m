clear all; close all; clc;
%%Least Squares - SVM

%%Loading the train test split random state 42 for the oversampled data
%%from what was done in python


X_train = csvread('.\Oversampled_Data\over_X_train.csv');
Y_train = csvread('.\Oversampled_Data\over_Y_train.csv');

X_test = csvread('.\Oversampled_Data\over_X_test.csv');
Y_test = csvread('.\Oversampled_Data\over_Y_test.csv');

X_train(1,:) = [];
X_train(:,1) = [];

X_test(1,:) = [];
X_test(:,1) = [];

Y_train(1,:) = [];
Y_train(:,1) = [];

Y_test(1,:) = [];
Y_test(:,1) = [];


%%Training a multiclass svm to evaluate TTT:
tic
multisvm(X_train,Y_train,X_test);
toc

%%Training the LS-SVM on the data


gam = 200;
sig2 = 20;
type = 'classification';

tic
[alpha,b] = trainlssvm({X_train,Y_train,type,gam,sig2,'RBF_kernel'});
toc
%%predicting 

Y_pred = simlssvm({X_train,Y_train,type,gam,sig2,'RBF_kernel'},{alpha,b},X_test);
Y_pred_proba = latentlssvm({X_train,Y_train,type,gam,sig2,'RBF_kernel'},{alpha,b},X_test);
%%Confusion matrices and recall method:

feature_order = [19 8 18 20 21 1 5 3 6 2 4 7 15 16 12 13 14 11 9 10 17];
indexes = [];
for i = 1:21
    indexes = [indexes;feature_order(i)];
    [alpha,b] = trainlssvm({X_train(:,indexes),Y_train,type,gam,sig2,'RBF_kernel'});
    Y_pred = simlssvm({X_train(:,indexes),Y_train,type,gam,sig2,'RBF_kernel'},{alpha,b},X_test(:,indexes));
    Y_pred_proba = latentlssvm({X_train(:,indexes),Y_train,type,gam,sig2,'RBF_kernel'},{alpha,b},X_test(:,indexes));
    csvwrite(strcat('lssvm_featureAnalysis_',num2str(i),'_pred.csv'),Y_pred);
    csvwrite(strcat('lssvm_featureAnalysis_',num2str(i),'_pred_proba.csv'),Y_pred);
end

confMatrix_1 = confusionmat(Y_test(:,1),Y_pred(:,1));
confMatrix_1
recall_1 = diag(confMatrix_1)./sum(confMatrix_1,1)';
recall_1

confMatrix_2 = confusionmat(Y_test(:,2),Y_pred(:,2));
confMatrix_2
recall_2 = diag(confMatrix_2)./sum(confMatrix_2,1)';
recall_2

confMatrix_3 = confusionmat(Y_test(:,3),Y_pred(:,3));
confMatrix_3
recall_3 = diag(confMatrix_3)./sum(confMatrix_3,1)';
recall_3


%%accuracy:

Y_pred_flattened = reshape(Y_pred.',1,[]); %%Need to flatten due to nature of confusionmat function
Y_test_flattened = reshape(Y_test.',1,[]);

confMatrix = confusionmat(Y_test_flattened,Y_pred_flattened);
accuracy = sum(diag(confMatrix))/sum(confMatrix(:));
accuracy

Eval = Evaluate(Y_test,Y_pred);
Eval

csvwrite('lssvm_proba_pred.csv',Y_pred_proba);

[area, se, thresholds, oneMinusSpec, sens, TN, TP, FN, FP] = roc(Y_test(:,1), Y_pred(:,1));

[tpr,fpr,thresholds] = roc(Y_pred_proba(:,3),Y_test(:,3));

specificity = 1- (FP/(FP+TN));
sensitivity = TP/(TP+TN);

%plot(oneMinusSpec,sens,'r'),xlabel("False Positive Rate"),ylabel("True Positive Rate");
plotroc(Y_test,Y_pred_proba,'r');
