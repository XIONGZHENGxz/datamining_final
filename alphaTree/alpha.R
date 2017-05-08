#!/user/bin/Rscript
library(ROCR) # make sure you installed this package
source("alphaTree.R") # change to the right path
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
print(cvatree[which.max(cvatree$avgPERF), ])
best_alpha = cvatree[which.max(cvatree$avgPERF), 'alpha']
best_depth = cvatree[which.max(cvatree$avgPERF), 'depth']

option <- list(max.depth = best_depth)

# build the optimal (need to adjust for the train and test data)
atfit <- atree(X_train, y_train, best_alpha, option)
output.json(atfit, model.name="alpha tree")
pred = prediction(predict(atfit, X_test)$minor.class, y_test)

# ROC and AUC
perf <- performance(pred,"tpr","fpr")
plot(perf, main = 'ROC curve')

perf1 <- performance(pred, 'auc')
auc = perf1@y.values

# lift chart
perf2 <- performance(pred, 'lift')
plot(perf2, main = 'Lift chart')
ind = which.max(slot(perf2, "y.values")[[1]])
lift = slot(perf2, "y.values")[[1]][ind]

# f1
perf3 <- performance(pred, 'f')
ind = which.max(slot(perf3, "y.values")[[1]])
f1 = slot(perf3, "y.values")[[1]][ind]

# accuracy
perf4 <- performance(pred, 'acc')
ind = which.max(slot(perf4, "y.values")[[1]])
acc = slot(perf4, "y.values")[[1]][ind]

