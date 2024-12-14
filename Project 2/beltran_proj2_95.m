close all; clear; clc
%% Loading the Data and storing them in a Table
[~,~,rawtrain] = xlsread('train.csv');
[~,~,rawtest] = xlsread('test.csv');
train1 = cell2table(rawtrain(2:end,:),'VariableNames',rawtrain(1,:));
test = cell2table(rawtest(2:end,:),'VariableNames',rawtest(1,:));
%% Accessing and Sample data cleaning
train_inputs = clean_data(train1);
stroke_response = train1.stroke;
test_inputs = clean_data(test);
%% Classification
[trainedClassifier, validationAccuracy] = trainClassifier(train_inputs, stroke_response)
stroke = trainedClassifier.predictFcn(test_inputs);
%% Sample Data Output
id = test.id;
% Preparing the Kaggle Submission
T=table(id, stroke);
writetable(T,'mySubmission.csv');
%% Function dataset accessing and cleaning
function [inputs] = clean_data(dataset)
    gender = dataset.gender;
    gender_cleaned(gender == "Male") = 1;
    gender_cleaned(gender == "Female") = 0;
    gender_cleaned = gender_cleaned';

    age = round(dataset.age);
    hypertension = dataset.hypertension;
    heartdisease = dataset.heart_disease;

    married = dataset.ever_married;
    married_cleaned(married == "Yes") = 1;
    married_cleaned(married == "No") = 0;
    married_cleaned = married_cleaned';

    worktype = dataset.work_type;
    worktype_cleaned(worktype == "Govt_job") = 1;
    worktype_cleaned(worktype == "Never_worked") = 2;
    worktype_cleaned(worktype == "Private") = 3;
    worktype_cleaned(worktype == "Self-employed") = 4;
    worktype_cleaned(worktype == "children") = 5;
    worktype_cleaned = worktype_cleaned';

    residencetype = dataset.Residence_type;
    residencetype_cleaned(residencetype == "Urban") = 1;
    residencetype_cleaned(residencetype == "Rural") = 0;
    residencetype_cleaned = residencetype_cleaned';

    avgglucoselvl = round(dataset.avg_glucose_level);
    bmi = dataset.bmi;
    bmi = round(fillmissing(bmi,'constant',0));

    smokingstatus = dataset.smoking_status;
    smokingstatus_cleaned(smokingstatus == "Unknown") = 1;
    smokingstatus_cleaned(smokingstatus == "formerly smoked") = 2;
    smokingstatus_cleaned(smokingstatus == "never smoked") = 3;
    smokingstatus_cleaned(smokingstatus == "smokes") = 4;
    smokingstatus_cleaned = smokingstatus_cleaned';

    inputs = [gender_cleaned age hypertension heartdisease ...
                    married_cleaned worktype_cleaned residencetype_cleaned ...
                    avgglucoselvl bmi smokingstatus_cleaned];
end
function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData, responseData)
    % Kaggle Score: 95.56%
    % Highest validationAccuracy = 
    % Holdout validation with 25% held out
    
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    % Convert input to table
    inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'});

    predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'};
    predictors = inputTable(:, predictorNames);
    response = responseData(:);
    isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.

    % Expand the Distribution Names per predictor
    % Numerical predictors are assigned either Gaussian or Kernel distribution and categorical predictors are assigned mvmn distribution
    % Gaussian is replaced with Normal when passing to the fitcnb function
    distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
    distributionNames(isCategoricalPredictor) = {'mvmn'};

    if any(strcmp(distributionNames,'Kernel'))
        classificationNaiveBayes = fitcnb(...
            predictors, ...
            response, ...
            'Kernel', 'Epanechnikov', ...
            'Support', 'Unbounded', ...
            'DistributionNames', distributionNames, ...
            'ClassNames', [0; 1]);
    else
        classificationNaiveBayes = fitcnb(...
            predictors, ...
            response, ...
            'DistributionNames', distributionNames, ...
            'ClassNames', [0; 1]);
    end

    % Create the result struct with predict function
    predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
    naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
    trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

    % Add additional fields to the result struct
    trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
    trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
    trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 10 columns because this model was trained using 10 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    % Convert input to table
    inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'});

    predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'};
    predictors = inputTable(:, predictorNames);
    response = responseData(:);
    isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

    % Set up holdout validation
    cvp = cvpartition(response, 'Holdout', 0.25);
    trainingPredictors = predictors(cvp.training, :);
    trainingResponse = response(cvp.training, :);
    trainingIsCategoricalPredictor = isCategoricalPredictor;

    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.

    % Expand the Distribution Names per predictor
    % Numerical predictors are assigned either Gaussian or Kernel distribution and categorical predictors are assigned mvmn distribution
    % Gaussian is replaced with Normal when passing to the fitcnb function
    distributionNames =  repmat({'Kernel'}, 1, length(trainingIsCategoricalPredictor));
    distributionNames(trainingIsCategoricalPredictor) = {'mvmn'};

    if any(strcmp(distributionNames,'Kernel'))
        classificationNaiveBayes = fitcnb(...
            trainingPredictors, ...
            trainingResponse, ...
            'Kernel', 'Epanechnikov', ...
            'Support', 'Unbounded', ...
            'DistributionNames', distributionNames, ...
            'ClassNames', [0; 1]);
    else
        classificationNaiveBayes = fitcnb(...
            trainingPredictors, ...
            trainingResponse, ...
            'DistributionNames', distributionNames, ...
            'ClassNames', [0; 1]);
    end

    % Create the result struct with predict function
    naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
    validationPredictFcn = @(x) naiveBayesPredictFcn(x);

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