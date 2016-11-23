setwd("C:\\Users\\lenovo\\Desktop\\R\\stat")
data <- read.table("data_filtered.csv",sep=',',header = T)
matrix = as.matrix(data)


length(which(data$biaoqian==1))

colsum1 = array(0,6457)
colsum2 = array(0,6457)
colsum3 = array(0,6457)
colsum4 = array(0,6457)
colsum5 = array(0,6457)
for (i in 1:3505){
  if (matrix[i,6458]==1){colsum1 = colsum1 + matrix[i,-6458]}
  if (matrix[i,6458]==2){colsum2 = colsum2 + matrix[i,-6458]}
  if (matrix[i,6458]==3){colsum3 = colsum3 + matrix[i,-6458]}
  if (matrix[i,6458]==4){colsum4 = colsum4 + matrix[i,-6458]}
  if (matrix[i,6458]==5){colsum5 = colsum5 + matrix[i,-6458]}
}
colsum = colSums(matrix[,-6458])
COLsum = as.data.frame(rbind(colsum1,colsum2,colsum3,colsum4,colsum5,colsum))
#length(which(colsum<(i+1)))-length(which(colsum<i))

once=which(colsum==1)
morethan_once=data[,-once]

twice=which(colsum<=2)
morethan_twice=data[,-twice]

three=which(colsum<=3)
morethan_3times=data[,-three]

ten=which(colsum<=10)
morethan_10times=data[,-ten]

library(randomForest)
set.seed(15)
# morethan_twice$biaoqian = as.factor(morethan_twice$biaoqian)
# attach(morethan_twice)
morethan_10times$biaoqian = as.factor(morethan_10times$biaoqian)
attach(morethan_10times)
# data$biaoqian = as.factor(data$biaoqian)
# attach(data)
ptm <- proc.time()
# rf.data = randomForest(biaoqian~.-biaoqian,data=morethan_twice, mtry=65,ntree=100)
 rf.data = randomForest(biaoqian~.-biaoqian,data=morethan_10times, mtry=100,ntree=100)
#rf.data = randomForest(biaoqian~.-biaoqian,data=data, mtry=100,ntree=100)
proc.time() - ptm
importance = rf.data$importance
n_features_used = length(which(importance>0))

#k-fold CV randomForest
#require(caret)
k = 5
#flds <- createFolds(morethan_10times, k = k, list =TRUE, returnTrain = FALSE)
CVData = morethan_10times[sample(3505),]
accuracy =array(0,5)
for (k in 1:5){
  test = CVData[((k-1)*701+1):(k*701),]
  train = CVData[-(((k-1)*701+1):(k*701)),]
  cvrf.data = randomForest(biaoqian~.-biaoqian,data=train, mtry=10,ntree=10)
  Y.test = predict(cvrf.data,newdata = test)
  accuracy[k] = length(which(Y.test == test[,ncol(test)]))/701
}
CV_accuracy = mean(accuracy)

#tune parameter
#mtry
mtryn1.accuracy =array(0,100)
test = CVData[1:701,]
train = CVData[-(1:701),]

for (i in 1:20){
  ntree = i*100
  cvrf.data = randomForest(biaoqian~.-biaoqian,data=train, mtry=18,ntree=ntree)
  Y.test = predict(cvrf.data,newdata = test)
  mtryn1.accuracy[i] = length(which(Y.test == test[,ncol(test)]))/701
}
  
plot(mtryn1.accuracy,type = "l")
points(mtryn1.accuracy,col = "red")
