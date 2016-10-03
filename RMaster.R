library(data.table)
library(xgboost)
library(caret)

# CARE: Both packages contain predict function
caretPredict <- caret::predict.preProcess
xgbPredict <- xgboost::predict
extPredict <- extraTrees::predict.extraTrees


tic <- Sys.time()

na.roughfix2 <- function (object, ...) {
  res <- lapply(object, roughfix)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x) {
  missing <- is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    x[missing] <- median.default(x[!missing])
  } else if (is.factor(x)) {
    freq <- table(x)
    x[missing] <- names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

# Set a seed
set.seed(2045558)

cat("reading the train and test data\n")
# Read train and test
dataTrain <- fread("train.csv", stringsAsFactors=TRUE) 
print(dim(dataTrain))
print(sapply(dataTrain, class))

dataSubmit <- fread("test.csv", stringsAsFactors=TRUE) 
print(dim(dataTrain))
print(sapply(dataTest, class))
cat("Data read ")
print(difftime( Sys.time(), start_time, units = 'sec'))


IDs <- dataTrain$ID
Y = dataTrain$target
X = as.data.frame(dataTrain)
X$target <- NULL
X$ID <- NULL

IDsSubmit <- dataSubmit$ID
YSubmit = dataSubmit$target
XSubmit = as.data.frame(dataSubmit)
XSubmit$ID <- NULL

# Combine for preprocessing
allData = rbind(X, XSubmit)


## Things from forum...

#  Result of Boruta, thanks to Florian
# https://www.kaggle.com/jimthompson/bnp-paribas-cardif-claims-management/using-the-boruta-package-to-determine-fe/discussion/comment/109207#post109207
#cat("Drop rejected vars - not important as found by Boruta\n")
#all_data$v72 <- NULL
#all_data$v62 <- NULL
#all_data$v112 <- NULL
#all_data$v107 <- NULL
#all_data$v125 <- NULL
#all_data$v75 <- NULL
#all_data$v71 <- NULL
#all_data$v91 <- NULL
#all_data$v74 <- NULL
#all_data$v52 <- NULL
#all_data$v22 <- NULL
#all_data$v3 <- NULL

m <- ncol(allData)
feature.names <- names(allData)

# make feature of counts of zeros factor
allData$ZeroCount <- rowSums(allData[,feature.names]== 0) / m
allData$Below0Count <- rowSums(allData[,feature.names] < 0) / m

# Idea from https://www.kaggle.com/sinaasappel/bnp-paribas-cardif-claims-management/exploring-paribas-data
#levels(all_data$v3)[1] <- NA #to remove the "" level and replace by NA
#levels(all_data$v22)[1] <- NA
#levels(all_data$v30)[1] <- NA
#levels(all_data$v31)[1] <- NA
#levels(all_data$v52)[1] <- NA
#levels(all_data$v56)[1] <- NA
#levels(all_data$v91)[1] <- NA
#levels(all_data$v107)[1] <- NA
#levels(all_data$v112)[1] <- NA
#levels(all_data$v113)[1] <- NA
#levels(all_data$v125)[1] <- NA

m <- ncol(allData)
feature.names <- names(allData)

for (c1 in 1:m)
{
  lvs = levels(allData[,c1])
  if (!is.null(lvs)) {
    if (lvs[1] == "") {
      levels(allData[,c1])[1] <-NA
      print(c("Replaced empty level with NA", feature.names[c1]))
    }
  }
}





#all_data$d115_69 <- all_data$v115 / all_data$v69
#all_data$d26_46 <- all_data$v26  / all_data$v46

# Convert factors to numerical
feature.names <- names(allData)
cat("assuming text variables are categorical & replacing them with numeric ids\n")
cat("re-factor categorical vars & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(allData[[f]])=="character" || class(allData[[f]])=="factor") 
    {
    allData[[f]] <- as.integer(factor(allData[[f]]))
  }
}



## Split back up
nTrain = dim(X)[1]
X = allData[1:nTrain,]
XSubmit = allData[(nTrain+1):dim(allData)[1],]

## Remove nearzerovar features with caret
feature.names <- names(X)
nzvX <- nearZeroVar(X, saveMetrics= TRUE, freqCut=95/5, uniqueCut=0.02)
# nzvXSubmit <- nearZeroVar(XSubmit, saveMetrics= TRUE, freqCut=95/5, uniqueCut=0.02)

# keepIdx = !(apply(rbind(nzvX$nzv, nzvXSubmit$nzv), 2, any))
keepIdx = !nzvX$nzv

X = X[, keepIdx]
XSubmit = XSubmit[, keepIdx]

print("Removed:")
print(feature.names[nzvX$nzv])



## Remove highly correlated features using caret
feature.names <- names(X)
# Note, can't use NACount_N here, as it's going to be 0 or unsuable (because row will contain a nan so won't be used)
# need to remove to avoid 0 std error on it (when calling)

numIdx = sapply(X, is.numeric) # now all numeric now anyway
XNum = X[,numIdx]
# instead, do this..
XNum = X
XNum$NACount_N <-NULL
# Actually, now this doesn't matter either. Moved to after.


descrCor <-  cor(x= XNum, use = "complete.obs")

highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .95)
hc <- findCorrelation(descrCor, cutoff = .95)

print("Removed:")
rmNames = names(XNum)[hc]
for (fn in 1:length(hc))
{
  X[rmNames[fn]] <- NULL
  XSubmit[rmNames[fn]] <- NULL
  print(rmNames[fn])
}

# X = X[,-hc]
# XSubmit = XSubmit[,-hc]

# From forum script - do they match?
#cat("Remove highly correlated features\n")
#highCorrRemovals <- c("v8","v23","v25","v36","v37","v46",
#                      "v51","v53","v54","v63","v73","v81",
#                      "v82","v89","v92","v95","v105","v107",
#                      "v108","v109","v116","v117","v118",
#                      "v119","v123","v124","v128")
# Just some Features to drop
#highCorrRemovals=c('v3','v22','v24','v30','v31','v47','v52','v56',
#            'v66','v71','v74','v75','v79','v91','v107','v110','v112',
#            'v113','v125')
#all_data <- all_data[,-which(names(all_data) %in% highCorrRemovals)]




# Small feature addition - Count NA percentage
m <- ncol(X)
X$NACount_N <- rowSums(is.na(X)) / m 
XSubmit$NACount_N <- rowSums(is.na(XSubmit)) / m 
feature.names <- names(X)


# Recombine, roughfix, find lin combos (no nas!) and then scale
n = dim(X)[1]
m = dim(X)[2]
allData = rbind(X, XSubmit)

# Replace empty values with median
for (i in 1:m)
{
  miss = is.na(allData[,i])
  if (any(miss))
  {
    allData[miss,i] = median(allData[!miss,i])
    print(c("Col: ", as.character(i), " Replacing na with ", as.character(median(allData[!miss,i]))))
  }
}
  

# lin combos
linCombos <- findLinearCombos(allData)
rm = linCombos$remove

if (!is.null(rm))
{
  X = X[, -rm]
  XSubmit = XSubmit[, -rm]
}

# scale
#allData = scale(allData)
# Calcaulte what to do just on X
preProVals <- preProcess(allData, method=c("center", "scale"))
# Apply to both X and XSubmit
allData <- caretPredict(preProVals, allData)


# split
X = allData[1:n,]
XSubmit = allData[(n+1):dim(allData)[1],]

print(dim(train))
#summary(train)tr
print(dim(test))
#summary(test)

#rm(all_data)
#gc()


# FEATURE SECTION OFF FOR NOW
#if (FALSE) {
#  # Boruta for feature selection used instead of ks
#  #Feature selection using KS test with 0.004 as cutoff.
#  tmpJ = 1:ncol(test)
#  ksMat = NULL
#  for (j in tmpJ) {
#    cat(j," ")
#    ksMat = rbind(ksMat, cbind(j, ks.test(train[,j],test[,j])$statistic))
#  }
#  
#  ksMat2 = ksMat[ksMat[,2]<0.007,]
#  feats = as.numeric(ksMat2[,1]) 
#  cat(length(feats),"\n")
#  cat(names(train)[feats],"\n")
#  var_to_drop <- setdiff(names(all_data), names(train)[feats])
#  cat("\nVars to drop:", var_to_drop, "\n")
#  # Input missing data & convert to xgb-data structure
#  #train[is.na(train)] <- -1
#  #test[is.na(test)] <- -1
#  
#  #xgtrain = xgb.DMatrix(as.matrix(train[,feats]), label = y, missing = -1)
#  #xgtest = xgb.DMatrix(as.matrix(test[,feats]), missing=-1)
#  
#  all_data <- rbind(train[,feats],test[,feats])
#}

# TAKING ROUGHFIX OUT FOR NOW
# all_data <- na.roughfix2(all_data)



## Scale and centre?



## Subset
n = dim(X)[1]
m = dim(X)[2]

rowInds = createDataPartition(Y, p=0.7, list=FALSE, times = 1)
#XTrain = X[rowInds,]
#YTrain = Y[rowInds]
# veto!
XTrain = X
YTrain = Y


rowInds = createDataPartition(Y, p=0.2, list=FALSE, times = 1)
XTest = X[rowInds,]
YTest = Y[rowInds]

rowInds = createDataPartition(Y, p=0.15, list=FALSE, times = 1)
XValid = X[rowInds,]
YValid = Y[rowInds]

# train <- all_data[1:n,]
# test <- all_data[(n+1):nrow(all_data),] 

xgtrain = xgb.DMatrix(as.matrix(XTrain), label = YTrain, missing=NA)
xgtest = xgb.DMatrix(as.matrix(XTest), missing=NA)
xgsubmit = xgb.DMatrix(as.matrix(XSubmit), missing=NA)

# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 2
    , data = xgtrain
    , early.stop.round = 10
    , maximize = FALSE
    , nthread = 6
  )
  gc()
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}

doTest <- function(param0, iter) {
  watchlist <- list('train' = xgtrain)
  model = xgb.train(
    nrounds = iter
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 20
    , nthread = 6
  )
  p <- predict(model, xgsubmit)
  rm(model)
  gc()
  p
}

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "eval_metric" = "logloss"
  , "eta" = 0.03
  , "subsample" = 0.9
  , "colsample_bytree" = 0.9
  , "min_child_weight" = 1
  , "max_depth" = 15
)

#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'sec'))
cat("Training a XGBoost classifier with cross-validation\n")
set.seed(2012823)
cv <- docv(param0, 500) 
# Show the clock
print( difftime( Sys.time(), start_time, units = 'sec'))

# sample submission total analysis
submission <- read.csv("sample_submission.csv")
ensemble <- rep(0, nrow(XTest))

cv <- round(cv * 1.8)
cat("Calculated rounds:", cv, " Starting ensemble\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results
its=5
ensStore = matrix(nrow=dim(XSubmit)[1], ncol = its)
for (i in 1:its) {
  print(i)
  set.seed(i + 168285)
  p <- doTest(param0, cv) 
  # use 40% to 50% more than the best iter rounds from your cross-fold number.
  # as you have another 50% training data now, which gives longer optimal training time
  ensemble <- ensemble + p
  ensStore[,i] = p
}

# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- ensemble/i

# Prepare submission
submission <- data.frame(ID=IDsSubmit, PredictedProb=ensemble/i)
cat("saving the submission file\n")
write.csv(submission, "submission.csv", row.names = F)

plot(ensemble/i)
histogram(ensemble/i)

# Stop the clock
#print(proc.time() - start_time)
print( difftime( Sys.time(), start_time, units = 'min'))

## Extra trees classifier
extTMod = extraTrees(XTrain, YTrain, numThreads = 6, ntree = 200)
setJavaMemory(12000)
predict(extTMod, newdata)

options( java.parameters = "-Xmx4g" )
library("extraTrees")

## Classification with ExtraTrees (with test data)
exTrain <- as.matrix(XTrain)
exTest <- as.matrix(XTest)
exSubmit <- as.matrix(XSubmit)
et <- extraTrees(exTrain, YTrain, ntree = 1000)
yhat <- predict(et, exTest)
yhat <- predict(et, exSubmit[1:500,])
## accuracy
mean(test$y == yhat)
## class probabilities
yprob = predict(et, test$x, probability=TRUE)
head(yprob)


# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- yhat

# Prepare submission
submission <- data.frame(ID=IDsSubmit, PredictedProb=yhat)
cat("saving the submission file\n")
write.csv(submission, "submission.csv", row.names = F)
