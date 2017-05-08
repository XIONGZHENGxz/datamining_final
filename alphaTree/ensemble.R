#!/user/bin/Rscript
library(ROCR) # make sure you installed this package
source("~/alphaTree.R") # change to the right path
lapply(list.files(pattern = "d3js/$"),source)

# data <- read.csv('training.csv') # check the path

# y <- data[ ,1]
# X <- data[ ,2:ncol(data)] 
y <- read.csv('y.csv', header = FALSE, sep = "" )
X <- read.csv('X.csv', header = FALSE, sep = "" )
data = cbind(y,X)

# split data into training and test
smp_size <- floor(0.67 * nrow(data))
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
X_train <- train[,2:ncol(train)]
y_train <- train[,1]
X_test <- test[,2:ncol(test)]
y_test <- test[,1]

## cross validation to choose the right alpha and depth

alphas = c(1,4,8,16,32)
depths = c(4,6,8,10,15)
cross_validation <- function(data, alpha, depth, fold = 5){
  folds <- cut(seq(1,nrow(data)),breaks=fold,labels=FALSE)
  
  pred_perf = matrix(,nrow = fold, ncol = length(alpha) * length(depth))
  
  #Perform 5 fold cross validation
  for (i in 1:fold){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    train_X = trainData[,2:ncol(trainData)]
    test_X = testData[,2:ncol(testData)]
    train_y = trainData[,1]
    test_y = testData[,1]
    
    for (k in 1: length(alpha)){
      for (j in 1: length(depth)){
        option <- list(max.depth = depth[j]) # specify tree depth
        atfit <- atree(train_X, train_y, alpha[k], option)
        pred = prediction(predict(atfit, test_X)$minor.class, test_y)
        
        perf1 <- performance(pred, 'auc')
        auc = perf1@y.values[[1]]
        
        # perf2 <- performance(pred, 'lift')
        # ind = which.max(slot(perf2, "y.values")[[1]])
        # lift = slot(perf2, "y.values")[[1]][ind]
        
        # perf3 <- performance(pred, 'f')
        # ind = which.max(slot(perf3, "y.values")[[1]])
        # f1 = slot(perf3, "y.values")[[1]][ind]
        
        # perf4 <- performance(pred, 'acc')
        # ind = which.max(slot(perf4, "y.values")[[1]])
        # acc = slot(perf4, "y.values")[[1]][ind]
        
        # measure = 0.3 * auc[[1]] + 0.3 * lift + 0.3 * f1 + 0.1 * acc
        
        pred_perf[i,k * j] = auc
      }
    }
    
  }
  results = data.frame('alpha' = rep(alpha, each = 5), 'depth' = rep(depth,5), 'avgPERF' = colMeans(pred_perf))
  return(results)
}

# need to adjust for the train data

cvatree = cross_validation(train, alphas, depths)
cvatree = cvatree[order(-cvatree$avgPERF),]

alpha1 = cvatree$alpha[1]
alpha2 = cvatree$alpha[2]
alpha3 = cvatree$alpha[3]

option1 <- list(max.depth = cvatree$depth[1])
option2 <- list(max.depth = cvatree$depth[2])
option3 <- list(max.depth = cvatree$depth[3])

# build the ensemble trees based on best parameters from CV (3 trees)
eat <- atree(X_train, y_train, alpha1, option1)
eat <- ensemble(eat, atree(X_train, y_train, alpha2, option2))
eat <- ensemble(eat, atree(X_train, y_train, alpha3, option3))
output.json(eat, model.name="ensemble 3 trees")
pred = prediction(predict(eat, X_test)$minor.class, y_test)

# ROC and AUC
perf <- performance(pred,"tpr","fpr")
plot(perf, main = 'ROC curve')

perf1 <- performance(pred, 'auc')
auc3 = perf1@y.values

# lift chart
perf2 <- performance(pred, 'lift')
plot(perf2, main = 'Lift chart')
ind = which.max(slot(perf2, "y.values")[[1]])
lift3 = slot(perf2, "y.values")[[1]][ind]

# f1
perf3 <- performance(pred, 'f')
ind = which.max(slot(perf3, "y.values")[[1]])
f13 = slot(perf3, "y.values")[[1]][ind]

# accuracy
perf4 <- performance(pred, 'acc')
ind = which.max(slot(perf4, "y.values")[[1]])
acc3 = slot(perf4, "y.values")[[1]][ind]


# build the ensemble trees based on best parameters from CV (6 trees)
alpha4 = cvatree$alpha[4]
alpha5 = cvatree$alpha[5]
alpha6 = cvatree$alpha[6]

option4 <- list(max.depth = cvatree$depth[4])
option5 <- list(max.depth = cvatree$depth[5])
option6 <- list(max.depth = cvatree$depth[6])

eat <- ensemble(eat, atree(X_train, y_train, alpha4, option4))
eat <- ensemble(eat, atree(X_train, y_train, alpha5, option5))
eat <- ensemble(eat, atree(X_train, y_train, alpha6, option6))
output.json(eat, model.name="ensemble 6 trees")
pred = prediction(predict(eat, X_test)$minor.class, y_test)

# ROC and AUC
perf <- performance(pred,"tpr","fpr")
plot(perf, main = 'ROC curve')

perf1 <- performance(pred, 'auc')
auc6 = perf1@y.values

# lift chart
perf2 <- performance(pred, 'lift')
plot(perf2, main = 'Lift chart')
ind = which.max(slot(perf2, "y.values")[[1]])
lift6 = slot(perf2, "y.values")[[1]][ind]

# f1
perf3 <- performance(pred, 'f')
ind = which.max(slot(perf3, "y.values")[[1]])
f16 = slot(perf3, "y.values")[[1]][ind]

# accuracy
perf4 <- performance(pred, 'acc')
ind = which.max(slot(perf4, "y.values")[[1]])
acc6 = slot(perf4, "y.values")[[1]][ind]

