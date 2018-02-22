---
layout: post
title:  "Practical Machine Learning with R"
date:   2018-02-21 23:00
category: r
icon: js
keywords: r, caret, coursera
image: 2.png
preview: 0
---

-   [Introduction](#introduction)
-   [Methodology](#methodology)
-   [1. Reading and Processing data](#reading-and-processing-data)
    -   [Remove Missing Values](#remove-missing-values)
    -   [Identify and eliminate highly correlated variables](#identify-and-eliminate-highly-correlated-variables)
-   [2. Subsetting Train/Test/Validation sets](#subsetting-traintestvalidation-sets)
-   [A note on Parallel Processing](#a-note-on-parallel-processing)
-   [3. Random Forest](#random-forest)
    -   [Random Forest Training Data Accuracy](#random-forest-training-data-accuracy)
    -   [Random Forrest Out of Sample Accuracy](#random-forrest-out-of-sample-accuracy)
-   [4. Generalized Boosting Model - GBM](#generalized-boosting-model---gbm)
    -   [Boosting Model - Training Data Accuracy](#boosting-model---training-data-accuracy)
    -   [GBM Out of Sample Accuracy](#gbm-out-of-sample-accuracy)
-   [5. Stacked Ensemble model](#stacked-ensemble-model)
    -   [Creating Combined model](#creating-combined-model)
    -   [Predicting on Validation set with the Combined Model](#predicting-on-validation-set-with-the-combined-model)
-   [Conclusion](#conclusion)

``` r
library(caret)
library(parallel)
library(doParallel)
```

### Introduction

This is a write-up for the course project for the [Johns Hopkins Practical Machine Learning MOOC on Coursera](https://www.coursera.org/learn/practical-machine-learning), which I completed in May of 2017. Since most of the heavy-lifting was already done, I chose this to be the inaugural post on my blog (work smarter, not harder).

The data set for this project is excercise data. For the project, I was tasked with building a model to predict whether or not an excercise was properly peformed, based on accelerometer data placed on the subjects. For anyone interested in following along, you can dowload both files by following these links : [training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv), [testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) - Note - this data set was used as a quiz for the course, and participants were required to score 20/20 to pass.

### Methodology

For my project, I chose to train a random forest model, a boosting model,
and a stacked ensemble model composed of the 2 afformentioned models. I used the caret package in R to train all of the models.

I chose to split my training data set into both a testing and validation data set, which is necessary for the ensemble model, to avoid issues due to over-fitting.

Additionally, I used the doParrallel package, which allows parallel processing in R on Windows machines, in order to speed up the training of my models.

1. Reading and Processing data
------------------------------

### Remove Missing Values

A quick inspection of the quiz data reveals numerous columns comprised of only NA values.
Although it is possible to impute these values, for simplicity's sake, I removed columns from both data sets where the quiz data had more than 90% missing values for that column.

Additionally, the first 6 columns in each data set contain information which is irrelevant to the quality of the excercise, so I will remove those features as well:

------------------------------------------------------------------------

``` r
# Remove columns (from full data set) with more than 90% missing values in quiz set
training_subset <- data[, colSums(is.na(quiz)) < nrow(quiz) * .90]

# For consistency sake, remove same columns from the quiz data
quiz_subset <- quiz[, colSums(is.na(quiz)) < nrow(quiz) * .90]

## remove columns that are not relevent to excercise quality
training_subset <- training_subset[,-(1:6)]
quiz_subset <- quiz_subset[,-(1:6)]

## check that column dimensions of data sets match
dim(training_subset)
```

    ## [1] 19622    53

``` r
dim(quiz_subset)
```

    ## [1] 20 53

------------------------------------------------------------------------

Encouragingly, both data sets have the same number of columns.

### Identify and eliminate highly correlated variables

It is important to identify and remove highly correlated variables in a data set, as including highly correlated variables in models can cause accuracy issues. Luckily, the caret package has a function, findCorrelation, which make identifying and removing these columns very easy. I chose to set the correlation cutoff at .9. I will use the training data to identify highly correlated variables, since the sample size is much larger than the quiz data.

------------------------------------------------------------------------

``` r
# create a correlation matrix 
corMatrix <- cor(training_subset[,-53])

# Create index with columns with high (>0.9) correlation
highCor <- findCorrelation(corMatrix, 0.9)

# Remove columns with correlation higher than .9 for both data sets
training_subset <- training_subset[, -highCor]
quiz_subset <- quiz_subset[,-highCor]
```

------------------------------------------------------------------------

2. Subsetting Train/Test/Validation sets
----------------------------------------

Using the caret package, I will subset my training data set into three parts:

-   training - the data used to train the models

-   testing - the data used to predict the out of sample accuracy for the individual models, and to create predictions for the ensemble model

-   validation - the data used to predict the out of sample accuracy for the stacked ensemble model

Note that for the data partitioning, I "set the seed", so that these results are reproducible.

------------------------------------------------------------------------

``` r
set.seed(808)
# subset data into validation and build subsets
inBuild <-createDataPartition(y = training_subset$classe, p = 0.7, list = F)

validation <- training_subset[-inBuild,]
buildData <- training_subset[inBuild,]

set.seed(818)
# Subset build data into train and test sets
inTrain <- createDataPartition(y = buildData$classe, p = 0.7, list = F)

training <- buildData[inTrain,]
testing <- buildData[-inTrain,]
```

------------------------------------------------------------------------

With the data separated into the appropriate sets, its time to train the models!

------------------------------------------------------------------------

A note on Parallel Processing
-----------------------------

In order to speed up the training of the models, I will utilize the the parallel and doParallel packages in R, which allow you to utilize multiple cores in training your models. Note that this process differs for Mac, so if you are working on that platform, you will want to consult other sources if you want to use parallel processing.

Before each I run each model (after having loaded both packages), I include the following lines of code:

``` r
# Initiate Parallel Processing 
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
```

After the code for the model, you will want to run the following lines, to stop the parallel processing.

``` r
stopCluster(cluster)
registerDoSEQ()
```

3. Random Forest
----------------

Although caret allows you to tune your models with various different paremeters, for this project I left all of the settings to default. For those interest in learning about the tuning parameters in caret, and I've found [this paper](https://www.jstatsoft.org/article/view/v028i05/v28i05.pdf) by Max Khun to be especially helpful in understanding some of the basic tuning paremeters for different types of models.

------------------------------------------------------------------------

``` r
# set seed for reproducibility
set.seed(828)
rfMod <- train(classe~., data = training, method="rf")

## test in-sample accuracy of model predictions
confusionMatrix(rfMod)
```

    ## Bootstrapped (25 reps) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 28.4  0.4  0.0  0.0  0.0
    ##          B  0.1 18.7  0.3  0.0  0.0
    ##          C  0.0  0.2 17.0  0.3  0.0
    ##          D  0.0  0.0  0.1 15.9  0.1
    ##          E  0.0  0.0  0.0  0.0 18.3
    ##                             
    ##  Accuracy (average) : 0.9821

------------------------------------------------------------------------

### Random Forest Training Data Accuracy

------------------------------------------------------------------------

The confusionMatrix command in the caret package allows us to see the accuracy of the newly trained model. We can see that the in sample accuracy for Random forest was 98.21%. While this is highly-accurate, we need to test the model on data that it was not trained on to get a realistic estimation of it's performance on the quiz data. This can also be done with the test data set, and confusionMatrix command in caret:

------------------------------------------------------------------------

### Random Forrest Out of Sample Accuracy

------------------------------------------------------------------------

``` r
# create list of prediction values on the testing set
rfPredict <- predict(rfMod, testing)

# pass table of prediction values vs actual to confusionMatrix
confusionMatrix(table(rfPredict, testing$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##          
    ## rfPredict    A    B    C    D    E
    ##         A 1170    3    0    0    0
    ##         B    0  792   10    0    0
    ##         C    0    2  704    5    2
    ##         D    0    0    4  669    2
    ##         E    1    0    0    1  753
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9927          
    ##                  95% CI : (0.9896, 0.9951)
    ##     No Information Rate : 0.2844          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9908          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9937   0.9805   0.9911   0.9947
    ## Specificity            0.9990   0.9970   0.9974   0.9983   0.9994
    ## Pos Pred Value         0.9974   0.9875   0.9874   0.9911   0.9974
    ## Neg Pred Value         0.9997   0.9985   0.9959   0.9983   0.9988
    ## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2841   0.1923   0.1710   0.1625   0.1829
    ## Detection Prevalence   0.2848   0.1948   0.1731   0.1639   0.1833
    ## Balanced Accuracy      0.9991   0.9954   0.9889   0.9947   0.9971

------------------------------------------------------------------------

The out of sample accuracy for Random Forrest is 99.27%, is actually *higher* than the training data, which is a surprising, yet encouraging result. Next we will train a boosting model, again using the caret package.

4. Generalized Boosting Model - GBM
-----------------------------------

------------------------------------------------------------------------

For my boosting model, I again chose to leave the tuning parameters to their default settings.

------------------------------------------------------------------------

``` r
## Set seed, train model and save into gbmMod
set.seed(838)
gbmMod <- train(classe~., method = "gbm", data = training, verbose = F)

## display accuracy of model on training data
confusionMatrix(gbmMod)
```

    ## Bootstrapped (25 reps) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 27.9  0.9  0.0  0.0  0.0
    ##          B  0.3 17.7  0.7  0.1  0.3
    ##          C  0.1  0.6 16.5  0.7  0.3
    ##          D  0.1  0.1  0.2 15.6  0.3
    ##          E  0.1  0.1  0.0  0.2 17.4
    ##                            
    ##  Accuracy (average) : 0.951

### Boosting Model - Training Data Accuracy

------------------------------------------------------------------------

The performance of the boosting model on the training data is nearly identical to random forest, only slightly better at 98.78%. We again need to validate our results using the testing data:

### GBM Out of Sample Accuracy

------------------------------------------------------------------------

``` r
## create list of preditions
gbmPredict <- predict(gbmMod, testing)

## use test-set predictions to create confusion matrix
confusionMatrix(table(gbmPredict, testing$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           
    ## gbmPredict    A    B    C    D    E
    ##          A 1154   24    0    0    0
    ##          B   11  745   34    4    2
    ##          C    2   28  672   27   13
    ##          D    3    0    8  640   10
    ##          E    1    0    4    4  732
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9575          
    ##                  95% CI : (0.9509, 0.9635)
    ##     No Information Rate : 0.2844          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9462          
    ##  Mcnemar's Test P-Value : 0.000122        
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9855   0.9348   0.9359   0.9481   0.9670
    ## Specificity            0.9919   0.9846   0.9794   0.9939   0.9973
    ## Pos Pred Value         0.9796   0.9359   0.9057   0.9682   0.9879
    ## Neg Pred Value         0.9942   0.9843   0.9864   0.9899   0.9926
    ## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2802   0.1809   0.1632   0.1554   0.1778
    ## Detection Prevalence   0.2861   0.1933   0.1802   0.1605   0.1799
    ## Balanced Accuracy      0.9887   0.9597   0.9577   0.9710   0.9821

------------------------------------------------------------------------

The accuracy for the boosting model on the test set was 99.2%, which leaves both models tied in terms accuracy on the test data.

5. Stacked Ensemble model
-------------------------

------------------------------------------------------------------------

The last model that I will train is a stacked ensemble model. This model will take the predictions of the random forest model and the boosting model, and use only those predictions as features to train on. Practically speaking, this was definitely overkill for this project, but it was something I was very interested in trying. :

### Creating Combined model

------------------------------------------------------------------------

In order to train the model, I must first create the data set, using the predictions I created on the test set from the models above. I combine those with the actual values for the "classe" variable in the test set, and I have a data set which I can now train my ensemble model:

``` r
# create data frame to stack boost/rf models, as well as the acual values of the "classe" variable
combined <- data.frame(rfPredict, gbmPredict, classe = testing$classe)

# Set seed an run random forest model using combined dataframe
set.seed(848)
comboMod <- train(classe~., method = "rf", data = combined, allowParallel = T)
```

### Predicting on Validation set with the Combined Model

------------------------------------------------------------------------

Since the ensemble model was trained on predictions from our test set, we need to use a different data set to predict the out of sample accuracy for the ensemble model. As you may remember, I created the validation set above, which we will now use to validate the data.

The key thing to remember with the ensemble model, is that it is not using the raw data set to predict the "classe" variable. The ensemble model is using the predictions of our random forest and boosting models as features. We therefore must first create a data frame of predictions from both models:

``` r
# Predict values for validation set using RF and GBM models from step 3
rfValPred <- predict(rfMod, validation)
gbmValPred <- predict(gbmMod, validation)

# Create dataframe with validation estimates for each model  
# **note that the column names must be exactly the same as the data that was used to train the ensemble model**
valDF <- data.frame(rfPredict = rfValPred, gbmPredict= gbmValPred)

# Use combined model to predict values for validation data frame
validationPredict <- predict(comboMod, valDF)

confusionMatrix(table(validationPredict, validation$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##                  
    ## validationPredict    A    B    C    D    E
    ##                 A 1672   10    0    0    0
    ##                 B    1 1123    9    0    0
    ##                 C    0    5 1013   15    2
    ##                 D    0    1    4  949    5
    ##                 E    1    0    0    0 1075
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.991           
    ##                  95% CI : (0.9882, 0.9932)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9886          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9860   0.9873   0.9844   0.9935
    ## Specificity            0.9976   0.9979   0.9955   0.9980   0.9998
    ## Pos Pred Value         0.9941   0.9912   0.9787   0.9896   0.9991
    ## Neg Pred Value         0.9995   0.9966   0.9973   0.9970   0.9985
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1908   0.1721   0.1613   0.1827
    ## Detection Prevalence   0.2858   0.1925   0.1759   0.1630   0.1828
    ## Balanced Accuracy      0.9982   0.9919   0.9914   0.9912   0.9967

------------------------------------------------------------------------

The combined model achieved 99.18% out of sample accuracy, while the diffence is incredibly slight, it appears that the ensemble model performed worse in terms of out of sample accuracy

Conclusion
----------

In this post I covered 3 different models which were used to predict the quality of an excercise, based on accelerometer data. Although I won't post the results of my predictions here (no cheaters!), my model did correctly predict the "classe" of exercise for all 20 of the subjects.
