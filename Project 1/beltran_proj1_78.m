close all; clear; clc
%% Loading the Data and storing them in a Table
[~,~,rawtrain] = xlsread('train.csv');
[~,~,rawtest] = xlsread('test.csv');
train1 = cell2table(rawtrain(2:end,:),'VariableNames',rawtrain(1,:));
test = cell2table(rawtest(2:end,:),'VariableNames',rawtest(1,:));
%% Accessing and Sample data cleaning
train_inputs = clean_data(train1);
survived = train1.Survived;
test_inputs = clean_data(test);
%% Classification
[trainedClassifier, validationAccuracy] = trainClassifier(train_inputs, survived)
Survived = trainedClassifier.predictFcn(test_inputs);
%% Sample Data Output
PassengerId = test.PassengerId;
% Preparing the Kaggle Submission
T=table(PassengerId, Survived);
writetable(T,'mysubmission.csv');
%% Function dataset accessing and cleaning
function [inputs] = clean_data(dataset)
    % Accessing dataset
    pclass = dataset.Pclass;
    sex = dataset.Sex;
    age = dataset.Age;
    sibsp = dataset.SibSp;
    parch = dataset.Parch;
    fare = dataset.Fare;
    embarked = dataset.Embarked;
    % Cleaning dataset
    sex_cleaned(dataset.Sex == "male") = 1;
    sex_cleaned(sex == "female") = 0;
    sex_cleaned = sex_cleaned';
    age = fillmissing(age,'constant',0);
    familysize = sibsp + parch + 1;
    fare = fix(fare); %% remove fractional part
    embarked_cleaned(string(embarked) == 'S') = 1;
    embarked_cleaned(string(embarked) == 'C') = 2;
    embarked_cleaned(string(embarked) == 'Q') = 3;
    embarked_cleaned(string(embarked) == ' ') = 4;
    embarked_cleaned = embarked_cleaned';
    inputs = [pclass sex_cleaned age familysize fare embarked_cleaned];
end
function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData, responseData)
    % Kaggle Score: 78.23/100
    % Highest validationAccuracy = 0.8643
    % Holdout validation with 29% held out
    
    % Extract predictors and response
    % This code processes the data into the right shape for training the model.
    % Convert input to table
    inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

    predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
    predictors = inputTable(:, predictorNames);
    response = responseData(:);
    isCategoricalPredictor = [false, false, false, false, false, false];

    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    classificationSVM = fitcsvm(predictors, response, ...
        'KernelFunction', 'gaussian', 'PolynomialOrder', [], ...
        'KernelScale', 2.4, 'BoxConstraint', 1, 'Standardize', true, ...
        'ClassNames', [0; 1]);

    % Create the result struct with predict function
    predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
    svmPredictFcn = @(x) predict(classificationSVM, x);
    trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

    % Add additional fields to the result struct
    trainedClassifier.ClassificationSVM = classificationSVM;
    trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
    trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 6 columns because this model was trained using 6 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

    % Extract predictors and response
    % This code processes the data into the right shape for training the model.
    % Convert input to table
    inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'});

    predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
    predictors = inputTable(:, predictorNames);
    response = responseData(:);
    isCategoricalPredictor = [false, false, false, false, false, false];

    % Set up holdout validation
    cvp = cvpartition(response, 'Holdout', 0.29);
    trainingPredictors = predictors(cvp.training, :);
    trainingResponse = response(cvp.training, :);
    trainingIsCategoricalPredictor = isCategoricalPredictor;

    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    classificationSVM = fitcsvm(trainingPredictors, trainingResponse, ...
        'KernelFunction', 'gaussian', 'PolynomialOrder', [], ...
        'KernelScale', 2.4, 'BoxConstraint', 1, ...
        'Standardize', true, 'ClassNames', [0; 1]);

    % Create the result struct with predict function
    svmPredictFcn = @(x) predict(classificationSVM, x);
    validationPredictFcn = @(x) svmPredictFcn(x);

    % Add additional fields to the result struct
    % Compute validation predictions
    validationPredictors = predictors(cvp.test, :);
    validationResponse = response(cvp.test, :);
    [validationPredictions, validationScores] = validationPredictFcn(validationPredictors);

    % Compute validation accuracy
    correctPredictions = (validationPredictions == validationResponse);
    isMissing = isnan(validationResponse);
    correctPredictions = correctPredictions(~isMissing);
    validationAccuracy = sum(correctPredictions)/length(correctPredictions);
    
    % Plot confusion matrix
    confmat = confusionmat(validationResponse,validationPredictions);
    confusionchart(confmat);
end