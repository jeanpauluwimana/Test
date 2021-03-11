# IST-707 Final Project
# Jean Paul Uwimana
# Date: 6/7/2019


# Loading required packages
library(tidyr)
library(dplyr)
library(tidytext)
library(stringr)
library(dplyr)
library(caret)
library(knitr)
library(FSelector)
library(RWeka)
library(ggplot2)
library(randomForest)

# Reading in the data
# heart <- read.csv("C:/Users/m316615/Documents/Personal-Temporary/IST-707/heart.csv")
heart <- read.csv(file.choose())
head(heart)

# Renaming age column
colnames(heart)[1] <- c("age")
# Variable importance
# importance <- FSelector::gain.ratio(target ~., data = heart) 
importance <- FSelector::gain.ratio(target ~., data = heart_data) 

# Pre-processing and removing non important variables
heart_data <- select(heart, c("cp", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"))

# Converting numeric variables to factors
heart_data$cp <- as.factor(heart_data$cp)
heart_data$exang <- as.factor(heart_data$exang)
heart_data$slope <- as.factor(heart_data$slope)
heart_data$ca <- as.factor(heart_data$ca)
heart_data$thal <- as.factor(heart_data$thal)
heart_data$target <- as.factor(heart_data$target)

#source("C:/Users/m316615/Documents/Personal-Temporary/IST-707/shuffle_cut.R")
source('C:/Users/Jpuwi/Documents/Syracuse_University/Portfolio2021/shuffle_cut.R')
heart_train <- shuffle_cut(data = heart_data, subset = "train")
heart_test <- shuffle_cut(data = heart_data, subset = "test")

# SVM Model
now <- Sys.time()
set.seed(100)
svm_heart_model <- caret::train(target ~., data = heart_train, method = "svmRadial",
                                na.action = na.omit,
                                preProcess = c("center", "scale"),
                                trControl = trainControl(method = "cv", number = 4),
                                tuneLength = 10)
time_elapsed <- Sys.time() - now
print(time_elapsed)

# Predicting heart disease presence/absence: SVM
svm_heart_pred <- predict(svm_heart_model, newdata = heart_test, type = "raw")
confusionMatrix(svm_heart_pred, heart_test$target, mode = "prec_recall")

# Naive Bayes Model
now <- Sys.time()
set.seed(100)
nb_heart_model <- caret::train(target ~., data = heart_train, method = "nb",
                               na.action = na.omit,
                               trControl = trainControl(method = "cv", number = 4),
                               tuneGrid = expand.grid(fL = 1,
                                                      usekernel = T, adjust = 1))
time_elapsed <- Sys.time() - now
print(time_elapsed)

# Predicting heart disease presence/absence: Naive Bayes
nb_heart_pred <- predict(nb_heart_model, newdata = heart_test, type = "raw")
confusionMatrix(nb_heart_pred, heart_test$target, mode = "prec_recall")

# RWeka's J48 - Decision Tree
now <- Sys.time()
set.seed(100)
J48_heart_model <- caret::train(target ~., data = heart_train, method = "J48",
                                na.action = na.omit,
                                trControl = trainControl(method = "cv", number = 4),
                                tuneLength = 10,
                                tuneGrid = expand.grid(M = 5, C = 0.005))
time_elapsed <- Sys.time() - now
print(time_elapsed)

# Predicting heart disease presence/abscence: J48
J48_heart_pred <- predict(J48_heart_model, newdata = heart_test, type = "raw")
confusionMatrix(J48_heart_pred, heart_test$target, mode = "prec_recall")

# Random Forest Model
now <- Sys.time()
set.seed(100)
rf_heart_model <- caret::train(target ~., data = heart_train, method = "rf",
                               na.action = na.omit,
                               trControl = trainControl(method = "cv", number = 4),
                               tuneLength = 10,
                               tuneGrid = expand.grid(mtry = 1))

time_elapsed <- Sys.time() - now
print(time_elapsed)
# Predicting heart disease presence/absence: Random Forest
rf_heart_pred <- predict(rf_heart_model, newdata = heart_test, type = "raw")
confusionMatrix(rf_heart_pred, heart_test$target, mode = "prec_recall")

# Model comparison
comparison <- caret::resamples(list(SVM = svm_heart_model, NB = nb_heart_model, J48 = J48_heart_model, RFM = rf_heart_model))

# Summary and Visualization of model comparison
summary(comparison)
bwplot(comparison)
