library(xgboost)
library(caret)
library(Metrics)
library(dplyr)
library(mlr)
library(Hmisc)
library(checkmate)


########### full data Cleaning ########### 
full <- read.csv("~/Desktop/STATS 101C/Final Project/lafdtraining updated.csv")
full$fd <- substring(full$incident.ID,1,7)
full<-full[which(!is.na(full$elapsed_time)),]
full$incident <- substring(full$incident.ID,9,length(full$incident.ID))
full$incident<-as.numeric(full$incident)
c <- full %>% group_by(incident.ID) %>% summarise(cnt=n())
full<-merge(full,c,by.x="incident.ID")
full$Incident.Creation.Time..GMT.<-as.numeric(full$Incident.Creation.Time..GMT.)
full$creation <- cut(full$Incident.Creation.Time..GMT., 
                     breaks = c(0, 6*60*60,12*60*60,18*60*60,
                                max(full$Incident.Creation.Time..GMT.)), 
                     labels = c(1:4))
full$elapsed_time <- log(full$elapsed_time)
full$incident.ID<-NULL
full$row.id<-NULL
full$Emergency.Dispatch.Code<-NULL
full$year<-as.factor(full$year)
full$First.in.District<-as.factor(full$First.in.District)
full$fd<-as.factor(full$fd)
full$Incident.Creation.Time..GMT. <- NULL



######################################
dmy <- dummyVars("~.", data=full)
trainTrsf <- data.frame(predict(dmy, newdata = full))
outcomeName <- c('elapsed_time')
predictors <- names(trainTrsf)[!names(trainTrsf) %in% outcomeName]

cv <- 5
trainSet <- trainTrsf
cvDivider <- floor(nrow(trainSet) / (cv+1))

smallestError <- 10000
for (depth in seq(1,10,1)) { 
  for (rounds in seq(1,20,1)) {
    totalError <- c()
    indexCount <- 1
    for (cv in seq(1:cv)) {
      # assign chunk to data test
      dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
      dataTest <- trainSet[dataTestIndex,]
      # everything else to train
      dataTrain <- trainSet[-dataTestIndex,]
      
      bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                     label = dataTrain[,outcomeName],
                     max.depth=3, nround=100,
                     objective = "reg:linear", verbose=0)
      gc()
      predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
      
      err <- rmse(as.numeric(dataTest[,outcomeName]), as.numeric(predictions))
      totalError <- c(totalError, err)
    }
    if (mean(totalError) < smallestError) {
      smallestError = mean(totalError)
      print(paste(depth,rounds,smallestError))
    }  
  }
} 
#param<-list(booster = "gblinear", objective = "reg:linear", eta=0.3, gamma=0, max_depth=6, min_child_weight=1,eval_metric="RMSE")
#xgbcv <- xgb.cv(params = param, data = as.matrix(trainSet[,predictors]), nrounds = 100, nfold = 10, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=3, nround=100, objective = "reg:linear", verbose=0) 

  ########### testing data Cleaning ########### 
testing <- read.csv("~/Desktop/STATS 101C/Final Project/testing.without.csv")
testing$elapsed_time<-0
testing$fd <- substring(testing$incident.ID,1,7)
testing<-testing[which(!is.na(testing$elapsed_time)),]
testing$incident <- substring(testing$incident.ID,9,length(testing$incident.ID))
testing$incident<-as.numeric(testing$incident)
c <- testing %>% group_by(incident.ID) %>% summarise(cnt=n())
testing<-merge(testing,c,by.x="incident.ID")
testing$Incident.Creation.Time..GMT.<-as.numeric(testing$Incident.Creation.Time..GMT.)
testing$creation <- cut(testing$Incident.Creation.Time..GMT., 
                        breaks = c(0,6*60*60,12*60*60,18*60*60,
                                   max(testing$Incident.Creation.Time..GMT.)), 
                        labels = c(1:4))
testing$incident.ID<-NULL
testing$row.id<-NULL
testing$Emergency.Dispatch.Code<-NULL
testing$year<-as.factor(testing$year)
testing$First.in.District<-as.factor(testing$First.in.District)
testing$fd<-as.factor(testing$fd)
testing$Incident.Creation.Time..GMT. <- NULL


fulltest<-rbind(full,testing)
#write.csv(fultest,file="fulltest.csv")

dmytest <- dummyVars("~.", data=fulltest)
testTrsf <- data.frame(predict(dmytest, newdata = fulltest))
pred <- predict(bst, as.matrix(testTrsf[,predictors]), outputmargin=TRUE)
# rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))
fulltest$pred<-pred
result<-fulltest[fulltest$elapsed_time==0,]
testing <- read.csv("~/Desktop/STATS 101C/Final Project/testing.without.csv")

final_result<-data.frame(row.id=testing$row.id,prediction=result$pred)
final_result$prediction <- exp(final_result$prediction)
write.csv(final_result,file="~/Desktop/exponential 3 100.csv",row.names=F)
