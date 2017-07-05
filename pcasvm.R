
library(dplyr) #dplyr provides a flexible grammar of data manipulation. It's the next iteration of plyr, focused on tools for working with data frames (hence the d in the name).
library(ggplot2) #for graphs
library(corrplot) #for correlation graph
library(gridExtra)
library(pROC) # Sample size / power computation
library(MASS) svmtools
library(caret)
library(caretEnsemble)
str(data)
data$diagnosis <- as.factor(data$diagnosis)
data$diagnosis
data[,33] <- NULL
summary(data)
#correlation of data
prop.table(table(data$diagnosis))
corr_mat <- cor(data[,3:ncol(data)])
corr_mat
library(corrplot)
corrplot(corr_mat) #correlation graph
set.seed(1234)
#divide into training and test data
data_index <- createDataPartition(data$diagnosis, p=0.7, list = FALSE)
train_data <- data[data_index, -1]
test_data <- data[-data_index, -1]
#for svm plots
col<-c("radius_mean","perimeter_mean","diagnosis")
col1<-c("radius_mean","area_mean","diagnosis")
col2<-c("radius_mean","radius_worst","diagnosis")
col3<-c("radius_mean","perimeter_worst","diagnosis")
col4<-c("radius_mean","area_mean","diagnosis")

train_data1 <- data[data_index,col]
test_data1 <- data[-data_index,col]
train_data2 <- data[data_index,col1]
test_data2<- data[-data_index,col1]
train_data3 <- data[data_index,col2]
test_data3 <- data[-data_index,col2]
train_data4 <- data[data_index,col3]
test_data4<- data[-data_index,col3]
train_data5 <- data[data_index,col4]
test_data5<- data[-data_index,col4]


library(e1071)#svm library
svmfit<-svm(diagnosis~.,data = train_data,kernel="linear",cost=.1,scale=F)
print(svmfit)#to find support vectors
# to plot svm graph of two quantities
svmfit1<-svm(diagnosis~.,data = train_data1,kernel="radial",cost=.1,scale=F)
svmfit2<-svm(diagnosis~.,data = train_data2,kernel="radial",cost=.1,scale=F)
svmfit3<-svm(diagnosis~.,data = train_data3,kernel="radial",cost=.1,scale=F)
svmfit4<-svm(diagnosis~.,data = train_data4,kernel="radial",cost=.1,scale=F)
svmfit5<-svm(diagnosis~.,data = train_data5,kernel="radial",cost=.1,scale=F)


plot(svmfit1,train_data[,col])
plot(svmfit2,train_data[,col1])
plot(svmfit3,train_data[,col2])
plot(svmfit4,train_data[,col3])
plot(svmfit5,train_data[,col4])
#to check which is best cost
tuned<-tune(svm,diagnosis~.,kernel="linear",data =train_data,ranges = list(cost=c(0.001,0.01,.1,1,10,100)))
summary(tuned)
#predict test data
pred_svm <- predict(svmfit, test_data)
pred_svm
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis, positive = "M")
cm_svm
#pca
pca_res <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(pca_res, type="l")
summary(pca_res)
pca_df <- as.data.frame(pca_res$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$diagnosis)) + geom_point(alpha=0.5)
fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)
#pca using svm
model_svm <- train(diagnosis~.,
                   train_data,
                   method="svmRadial",
                   metric="ROC",
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)
pred_svm <- predict(model_svm, test_data)
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis, positive = "M")
cm_svm
model_lda <- train(diagnosis~.,
                   train_data,
                   method="lda2",
                   #tuneLength = 10,
                   metric="ROC",
                   preProc = c("center", "scale"),
                   trControl=fitControl)
pred_lda <- predict(model_lda, test_data)
cm_lda <- confusionMatrix(pred_lda, test_data$diagnosis, positive = "M")
cm_lda
model_rf1 <- train(diagnosis~.,
                  train_data,
                  method="ranger",
                  metric="ROC",
                  #tuneLength=10,
                  #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)
pred_rf <- predict(model_rf, test_data)
cm_rf <- confusionMatrix(pred_rf, test_data$diagnosis, positive = "M")
cm_rf
model_pca_rf <- train(diagnosis~.,
                      train_data,
                      method="ranger",
                      metric="ROC",
                      #tuneLength=10,
                      #tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                      preProcess = c('center', 'scale', 'pca'),
                      trControl=fitControl)
pred_pca_rf <- predict(model_pca_rf, test_data)
cm_pca_rf <- confusionMatrix(pred_pca_rf, test_data$diagnosis, positive = "M")
cm_pca_rf
model_knn <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)
pred_knn <- predict(model_knn, test_data)
cm_knn <- confusionMatrix(pred_knn, test_data$diagnosis, positive = "M")
cm_knn
library(nnet)
model_nnet <- train(diagnosis~.,
                    train_data1,
                    method="nnet",
                    metric="ROC",
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    tuneLength=10,
                    trControl=fitControl)
pred_nnet <- predict(model_nnet, test_data)
cm_nnet <- confusionMatrix(pred_nnet, test_data$diagnosis, positive = "M")
cm_nnet
model_nb <- train(diagnosis~.,
                  train_data,
                  method="nb",
                  metric="ROC",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)
pred_nb <- predict(model_nb, test_data)
cm_nb <- confusionMatrix(pred_nb, test_data$diagnosis, positive = "M")
cm_nb
model_pca_nnet <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
pred_pca_nnet <- predict(model_pca_nnet, test_data)
cm_pca_nnet <- confusionMatrix(pred_pca_nnet, test_data$diagnosis, positive = "M")
cm_pca_nnet
model_lda_nnet <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
pred_lda_nnet <- predict(model_lda_nnet, test_data)
cm_lda_nnet <- confusionMatrix(pred_lda_nnet, test_data$diagnosis, positive = "M")
cm_lda_nnet
model_pca_nnet <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
pred_pca_nnet <- predict(model_pca_nnet, test_data)
cm_pca_nnet <- confusionMatrix(pred_pca_nnet, test_data$diagnosis, positive = "M")
cm_pca_nnet
model_list <- list( NNET=model_nnet, PCA_NNET=model_pca_nnet, LDA_NNET=model_lda_nnet, 
                   KNN = model_knn,SVM=model_svm, NB=model_nb)
resamples <- resamples(model_list)
model_cor <- modelCor(resamples)
corrplot(model_cor)
bwplot(resamples, metric="ROC")
cm_list <- list( NNET=cm_nnet, PCA_NNET=cm_pca_nnet, LDA_NNET=cm_lda_nnet, 
                KNN = cm_knn,SVM=model_svm, NB=cm_nb)
dim(x)
cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results
cm_results_max <- apply(cm_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report

#parallelSVM
library(parallelSVM)
#kernel=radial
serialSvm <- svm(diagnosis~.,data = train_data,kernel="radial",cost=.1,scale=F)
parallelSvm <- parallelSVM(diagnosis ~ ., data = train_data,numberCores = 5
                                  , samplingSize = 0.7,
                                       probability = TRUE, gamma=0.1, cost = 10)
# Calculate predictions
serialPredictions <- predict(serialSvm, test_data)
cm_svm <- confusionMatrix(serialPredictions, test_data$diagnosis, positive = "M")
cm_svm
parallelPredicitions <- predict(parallelSvm, test_data)
sm_svm <- confusionMatrix(parallelPredicitions, test_data$diagnosis, positive = "M")
sm_svm

#time complexity
system.time(serialSvm   <- svm(diagnosis ~ ., data=train_data,kernel="radial", 
                               probability=TRUE, cost=10, gamma=0.1))
system.time(parallelSvm <- parallelSVM(diagnosis ~ ., data = train_data[,-1],
                                       numberCores = 8, samplingSize = 0.2, 
                                       probability = TRUE, gamma=0.1, cost = 10))

# Calculate predictions
system.time(serialPredictions <- predict(serialSvm, test_data))
system.time(parallelPredicitions <- predict(parallelSvm, test_data))

