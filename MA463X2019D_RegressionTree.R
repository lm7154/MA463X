#getwd()
#setwd("C:/Users/Elizabeth Dan Mao/Downloads")
#Latest Update: 04/22/2019
#Creator: Dan Mao
#Purpose: MA463X-2019-DTerm Project
#Data Source = 'https://www.kaggle.com/c/house-prices-advanced-regression-techniques'
#Teammate:
#Reference = c('http://r-statistics.co/Outlier-Treatment-With-R.html','https://educationalresearchtechniques.com/2017/05/22/9121/')

#Set up dir
#getwd()
#setwd("...") <-- Your working dir should be here
#----------------------------------------------------------------------------------------------------------
# import library
library(tree)
library(lattice)
library(ggplot2)
library(caret)
library(gbm)
library(corrplot)


#read house training and testing data from the csv file
#NOTICE that we dont need to call data.frame since it returns a data frame
house.train = read.csv('DanTrainData.csv')
house.test = read.csv('DanTestData.csv')

#---------------------------------------------------------------------------------------
#Define R^2 function
rsq = function(x,y) cor(x,y)^2

#---------------------------------------------------------------------------------------
#filter outliers
mod = lm(SalePrice ~ ., data=house.train)
cooksd = cooks.distance(mod)
#Visualize the plot
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="blue")

#summary(cooksd)
influential = as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
influential
#manually delete following data points 214  314  450  472  609  892  922 1082

#---------------------------------------------------------------------------------------
#Get a brief sense of how the dataset looks like
#Histogram of SalePrice
hist(house.train$SalePrice)

#----------------------------------------------------------------------------------------
#MODEL 1
tree.model = tree(SalePrice~., data = house.train)
plot(tree.model)
text(tree.model, pretty = 1)

#predicting, and draw a plot of error
pred = predict(tree.model, house.test)
resid = house.test$SalePrice-pred

hist(resid)
plot(pred.2~house.test$SalePrice);abline(0,1,col = 'blue')
rsq(pred.2, house.test$SalePrice)

#----------------------------------------------------------------------------------------------------------------
#MODEL 2
#boosting tree

#Cross-validation for tree
grid = expand.grid(.n.trees=seq(200,800,by=200),.interaction.depth=seq(2,6,by=1),.shrinkage=c(.001,.01,.1),
                  .n.minobsinnode=10)
control = trainControl(method = "CV")
set.seed(10)
gbm.train = train(SalePrice~.,data=house.train,method='gbm',trControl=control,tuneGrid=grid)
gbm.train

#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were n.trees = 800, interaction.depth = 6, shrinkage = 0.01 and n.minobsinnode = 10.
set.seed(20)
gbm.price = gbm(SalePrice~.,data=house.train,n.trees = 800,interaction.depth = 6,
               shrinkage =0.01,distribution = 'gaussian')
#try some other coefficient
gbm.price.3 = gbm(SalePrice~.,data=house.train,n.trees = 1000,interaction.depth = 7,
                  shrinkage =0.01,distribution = 'gaussian')

#test the model
gbm.test = predict(gbm.price,newdata = house.test,n.trees = 800)
gbm.test.3  =  predict(gbm.price.3,newdata = house.test,n.trees = 1000)
gbm.resid = gbm.test-house.test$SalePrice

hist(gbm.resid)
mean(gbm.resid^2)
plot(gbm.test,house.test$SalePrice);abline(0,1,col = 'blue')
rsq(gbm.test, house.test$SalePrice)
rsq(gbm.test.3, house.test$SalePrice)

#generate test for Kaggle for model 2
kaggle.test= read.csv("test_out.csv")
kaggle.result.1 = predict(gbm.price,newdata = kaggle.test,n.trees = 800)
write.csv(kaggle.result.1, file = "kaggleResult.csv")

#---------------------------------------------------------------------------------------
#Model 3: Apply boosting tree model, after filtering outliers
#Read in a new .csv fil that don't contains outliers that we got using Cooks distance. 
house.train.clean = read.csv('DanTrainData_filtered.csv')

#Cross Vliadation, WARNING: Time-cosuming, Just call gbm.train.clean
grid = expand.grid(.n.trees=seq(200,1200,by=200),.interaction.depth=seq(4,8,by=1),.shrinkage=c(.001,.01,.1),
                   .n.minobsinnode=10)
control = trainControl(method = "CV")
set.seed(30)
gbm.train.clean = train(SalePrice~.,data=house.train.clean,method='gbm',trControl=control,tuneGrid=grid)
gbm.train.clean

#Another Cross-Validation, WARNING: Time-cosuming, Just call gbm.train.cleanAlt
gridAlt = expand.grid(.n.trees=seq(800,1400,by=200),.interaction.depth=seq(6,10,by=1),.shrinkage=c(.001,.01,.1),
                   .n.minobsinnode=10)
gbm.train.cleanAlt = train(SalePrice~.,data=house.train.clean,method='gbm',trControl=control,tuneGrid=gridAlt)
gbm.train.cleanAlt
gbm.price.cleanAlt = gbm(SalePrice~.,data=house.train.clean,n.trees = 1200,interaction.depth = 6,
                      shrinkage =0.01,distribution = 'gaussian')

#gbm.train.clean
#The final values used for the model were n.trees = 1000, interaction.depth = 7, shrinkage = 0.01 and n.minobsinnode = 10.
gbm.price.clean = gbm(SalePrice~.,data=house.train.clean,n.trees = 1000,interaction.depth = 7,
                shrinkage =0.01,distribution = 'gaussian')

#Test model:gbm.test.clean
gbm.test.clean = predict(gbm.price.clean,newdata = house.test,n.trees = 1000)
rsq(gbm.test.clean, house.test$SalePrice)
plot(gbm.test.clean,house.test$SalePrice);abline(0,1,col = 'blue')
#Kaggle.result
kaggle.result.2 = predict(gbm.price.clean,newdata = kaggle.test,n.trees = 1000)
write.csv(kaggle.result.2, file = "kaggleResult_v2.csv")

#Test model:gbm.test.cleanAlt
gbm.test.cleanAlt = predict(gbm.price.cleanAlt,newdata = house.test,n.trees = 1200)
rsq(gbm.test.cleanAlt, house.test$SalePrice)
plot(gbm.test.cleanAlt,house.test$SalePrice);abline(0,1,col = 'blue')
kaggle.result.3 = predict(gbm.price.cleanAlt,newdata = kaggle.test,n.trees = 1200)
write.csv(kaggle.result.3, file = "kaggleResult_v3.csv")

#-----------------------------------------------------------------------------------------------------------------
#conclusion: R^2 Plot
#y = c(0.91054, 0.91585, 0.91605, 0.7006614, 0.9067327, 0.9090176, 0.89604, 0.8324, 0.90151, 0.89688, 0.89181, 0.88321, 0.87634)
#x = c('Ridge', 'Lasso', 'Elastic Net', 'Regression Tree', 'Boosting Tree', 'Updated Boosting Tree', 'Random Forest(15)',
#      'Random Forest(2)', 'Random Forest(46)', 'Random Forest(90)', 'Random Forest(134)', 'Random Forest(178)', 'Random Forest(222)')
#plot(y~x)
#result = array(c(x,y),)
#print(result)
