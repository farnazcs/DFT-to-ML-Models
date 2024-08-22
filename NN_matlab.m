clc
clear all
close all
% reading data
[d1,s1] = xlsread('C:\Users\farno\Downloads\NData\data\NM.csv');
[d2,s2] = xlsread('C:\Users\farno\Downloads\NData\data\NF.csv');
[d3,s3] = xlsread('C:\Users\farno\Downloads\NData\data\NC.csv');
[d4,s4] = xlsread('C:\Users\farno\Downloads\NData\data\MF.csv');
[d5,s5] = xlsread('C:\Users\farno\Downloads\NData\data\MC.csv');
[d6,s6] = xlsread('C:\Users\farno\Downloads\NData\data\FC.csv');
BW=xlsread('C:\Users\farno\Downloads\NData\data\DB_bw_v2.csv');

AllD=[d1;d2;d3;d4;d5;d6];
PlotC=12;
% picking the important variables from the file
DM=[d1(:,4);d2(:,4);d3(:,4);d4(:,4);d5(:,4);d6(:,4)];
DE=[d1(:,PlotC);d2(:,PlotC);d3(:,PlotC);d4(:,PlotC);d5(:,PlotC);d6(:,PlotC)];
histogram(DM)
histogram(DE(DE~=0),20)

% normalizing the variables for better training
for i=1:190
    Dtrain(i,AllD(i,25)-24)=AllD(i,27)/8;
    Dtrain(i,4+AllD(i,26)-24)=AllD(i,28)/16;
end

edges = linspace(0, 750, 21); 
histogram(AllD(:,end),'BinEdges',edges)


% randomly shuffling the train data to remove any correlation between
% training data samples
AllX=Dtrain;%each column is one data
AllR=AllD(:,end);
RP=randperm(size(AllX, 1));
AllX = AllX(RP, :);
AllR = AllR(RP, :);

% dividing the data to training and validation data
XT=AllX(1:end-30,:);
RT=AllR(1:end-30,:);

XV=AllX(end-30+1:end,:);
RV=AllR(end-30+1:end,:);

% defining our fully-connected (feedforward neural network layers)
NS=30;
layers = [ ...
    featureInputLayer(8, "Name", "myFeatureInputLayer") % , 'Normalization','rescale-symmetric'
    fullyConnectedLayer(30, "Name", "myFullyConnectedLayer1") % 8 neurons, fully connected
    batchNormalizationLayer
    %tanhLayer("Name", "myTanhLayer") % Activation function: Hyperbolic tangent
    %sigmoidLayer("Name", "SigmoidLayer")
    reluLayer("Name", "reluLayer")
    %functionLayer(@(X) exp(-0.5*X.^2),Description="softsign")
    %dropoutLayer(0.5)

    %fullyConnectedLayer(NS, "Name", "myFullyConnectedLayer2") % Output layer (1 neuron)
    %batchNormalizationLayer
    %functionLayer(@(X) exp(-0.5*X.^2),Description="softsign")
    %reluLayer("Name", "reluLayer2")
    fullyConnectedLayer(1, "Name", "myFullyConnectedLayer3") % Output layer (1 neuron)
    regressionLayer("Name", "myRegressionLayer") % Regression layer for output layer
    ];

% defining the network training options

opts = trainingOptions('adam', ...            % Optimizer = 'adam'
    'ExecutionEnvironment','gpu',...
    'MaxEpochs',1000,  ...                    % Maximum number of epochs
    'InitialLearnRate',0.01, ...              % Learning rate
    'Shuffle','every-epoch', ...              % Shuffle data every epoch
    'Plots','training-progress', ...           % Display training progress
     'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',300, ...
    'MiniBatchSize',64, ...                  % Batch size
    'Verbose',false,...
     'ValidationData',{XV,RV}, ...
    'ValidationFrequency',30);   


% train the network using the training data
[trainedNet, info] = trainNetwork(XT, RT, layers, opts);
% checking the trainned network on validation data
YTest = predict(trainedNet, XV);      % Pass the test points through the trained network and get the predicted outputs

plot(RV)
hold on
plot(YTest)
figure 
plot(RV,YTest,'.')
xlim([0,500])
ylim([0,500])


% now doing random forest: https://www.analytixlabs.co.in/blog/random-forest-regression/#:~:text=A%20Random%20forest%20regression%20model,all%20the%20individual%20trees'%20predictions.
%defining and creating the random forest 
tree = fitrtree(XT,RT,...
                'CategoricalPredictors',8,'MinParentSize',15)
% validationg the tree on test data
YTest = predict(tree,XV);
figure
plot(RV)
hold on
plot(YTest)
 
figure
plot(RV,YTest,'.')
xlim([0,500])
ylim([0,500])

% fit SVM: Support Vector Machine
tree = fitrsvm(XT,RT,'Standardize',true,'KernelFunction','polynomial');
tree.ConvergenceInfo.Converged
tree.NumIterations
YTest = predict(tree,XV);
figure
plot(RV)
hold on
plot(YTest)
 
figure
plot(RV,YTest,'.')
xlim([0,500])
ylim([0,500])
