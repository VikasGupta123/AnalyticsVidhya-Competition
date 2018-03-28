library(rpart)
library(rpart.plot)
library(e1071)
library(caret)
library(randomForest)
library(mice)
library(xgboost)
library(glmnet)
library(Metrics)
#reading data

Train= read.csv("train.csv", na.strings = "")
Test= read.csv("test.csv", na.strings = "")


y= Train$SALES_PRICE

Train$SALES_PRICE= NULL
total= rbind(Train, Test)
total$PRT_ID=NULL



# checking missing values
colSums(is.na(total))

#imputing missing values
total$N_BATHROOM[is.na(total$N_BATHROOM)] =median(total$N_BATHROOM, na.rm = T)
total$QS_OVERALL[is.na(total$QS_OVERALL)]=median(total$QS_OVERALL, na.rm = T)
total$N_BEDROOM[is.na(total$N_BEDROOM)]=median(total$N_BEDROOM, na.rm = T)


total$DATE_SALE= as.Date(total$DATE_SALE, "%d-%m-%Y")
total$DATE_BUILD= as.Date(total$DATE_BUILD, "%d-%m-%Y")

total$AGE= total$DATE_SALE-total$DATE_BUILD
total$AGE= as.integer(total$AGE)

total$DATE_SALE=substring(as.character(total$DATE_SALE),1,4)
total$DATE_SALE= as.integer(total$DATE_SALE)
total$DATE_BUILD=substring(as.character(total$DATE_BUILD),1,4)
total$DATE_BUILD= as.integer(total$DATE_BUILD)

total$AREA[total$AREA=="Adyr"]= "Adyar"
total$AREA=factor(total$AREA)
total$AREA[total$AREA=="Ana Nagar"| total$AREA=="Ann Nagar"]= "Anna Nagar"
total$AREA= factor(total$AREA)
total$AREA[total$AREA=="Chormpet"| total$AREA=="Chrmpet"|total$AREA=="Chrompt"]= "Chrompet"
total$AREA= factor(total$AREA)
total$AREA[total$AREA=="Karapakam"]= "Karapakkam"
total$AREA=factor(total$AREA)
total$AREA[total$AREA=="KKNagar"]= "KK Nagar"
total$AREA=factor(total$AREA)
total$AREA[total$AREA=="TNagar"]= "T Nagar"
total$AREA=factor(total$AREA)
total$AREA[total$AREA=="Velchery"]= "Velachery"
total$AREA=factor(total$AREA)


####
total$SALE_COND[total$SALE_COND=="Ab Normal"]= "AbNormal"
total$SALE_COND=factor(total$SALE_COND)
total$SALE_COND[total$SALE_COND=="Adj Land"]= "AdjLand"
total$SALE_COND=factor(total$SALE_COND)
total$SALE_COND[total$SALE_COND=="Partiall"|total$SALE_COND=="PartiaLl"]= "Partial"
total$SALE_COND=factor(total$SALE_COND)

#######

total$PARK_FACIL[total$PARK_FACIL=="Noo"]= "No"
total$PARK_FACIL=factor(total$PARK_FACIL)
######

total$BUILDTYPE[total$BUILDTYPE=="Comercial"|total$BUILDTYPE=="Commercil"]= "Commercial"
total$BUILDTYPE=factor(total$BUILDTYPE)

total$BUILDTYPE[total$BUILDTYPE=="Other"]= "Others"
total$BUILDTYPE=factor(total$BUILDTYPE)
#####

total$UTILITY_AVAIL[total$UTILITY_AVAIL=="All Pub"]= "AllPub"
total$UTILITY_AVAIL=factor(total$UTILITY_AVAIL)
######
total$STREET[total$STREET=="NoAccess"]= "No Access"
total$STREET=factor(total$STREET)
total$STREET[total$STREET=="Pavd"]= "Paved"
total$STREET=factor(total$STREET)
######

total$N_ROOMS= total$N_BEDROOM + total$N_BATHROOM + total$N_ROOM
total$N_BEDROOM= NULL
total$N_BATHROOM=NULL
total$N_ROOM= NULL
total$QS_ROOMS= NULL
total$QS_BATHROOM= NULL
total$QS_BEDROOM= NULL
total$DATE_BUILD= NULL
total$REG_FEE=NULL
total$COMMIS=NULL
total$DIST_MAINROAD=NULL
total$DATE_SALE=NULL

#######


Train= total[1:7109,]
Train1=Train
Train1$SALES_PRICE=y
Test= total[7110:10034,]
Train_xgb= sparse.model.matrix(~.-1,Train)
Test_xgb= sparse.model.matrix(~.-1,Test)
bstcv=xgb.cv(data = Train_xgb, nrounds = 4000, nfold = 4, label = y, 
             eta = 0.1, max.depth = 3, objective = "reg:linear", subsample = 0.8)


Train_xgb1=Train[1:4000,]
Train_xgb1= sparse.model.matrix(~.-1,Train_xgb1)
dtrain= Train_xgb1
dtrain= as.matrix(dtrain)
dtrain <- xgb.DMatrix(data = dtrain, label=y[1:4000])

Test_xgb1= Train[4001:7109,]
Test_xgb1= sparse.model.matrix(~.-1,Test_xgb1)
dtest= Test_xgb1
dtest= as.matrix(dtest)
dtest <- xgb.DMatrix(data = dtest, label=y[4001:7109])

watchlist <- list(train=dtrain, test=dtest)

smallestError <- 5000000
for (et in c(0.05,0.1,0.15,0.2,0.25,0.3)){
  for (depth in seq(1,7,1)) {
    # train
    bst <- xgb.train(data=dtrain, max.depth=2, eta=et, nround=500, watchlist=watchlist, verbose = 0)
    pred <- predict(bst, dtest)
    err <- rmse(y[4001:7109], pred)
    
    if (err < smallestError) {
      smallestError = err
      print(paste(et,depth,err))
    }
    
     
  }
}



smallestError <- 5000000
for (et in c(0.05,0.1,0.15,0.2,0.25,0.3)){
 for (depth in seq(1,7,1)) {
     # train
     bst <- xgboost(data = Train_xgb1,
                   label = as.numeric(log(y[1:4000])),eta= et,
                   max.depth=depth, nround=500, verbose=0)
     
    
     # predict
     pred <- predict(bst, Test_xgb1)
     err <- rmse(log(y[4001:7109]), pred)
    
     if (err < smallestError) {
       smallestError = err
       print(paste(et,depth,err))
    }     
  }
}

bstdense <- xgboost(data = Train_xgb1,label = y[1:4000],eta= 0.3,max.depth=3, nround=500, verbose = 0)
######
xgbGrid <- expand.grid(
  nrounds = c(10000),
  max_depth = seq(3,6,by=1),
  eta = seq(0.03,0.05,by=0.01),
  gamma = seq(0,1,by=1),
  colsample_bytree = seq(0.4,0.6,by = 0.1),
  min_child_weight = seq(1,1,by = 0.5),
  subsample = seq(0.4,0.6,by = 0.1)
)

rmseErrorsHyperparameters <- apply(xgbGrid, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentMin_Child_Weight <- parameterList[["min_child_weight"]]
  currentGamma <- parameterList[["gamma"]]
  currentEta <- parameterList[["eta"]]
  currentMax_Depth <- parameterList[["max_depth"]]
  currentNrounds <- parameterList[["nrounds"]]
  
  params <- list(objective = "reg:linear", 
                 #booster = "gbtree", 
                 #eta = 2/currentNrounds,
                 eta = currentEta, 
                 gamma = currentGamma, 
                 max_depth = currentMax_Depth, 
                 min_child_weight = currentMin_Child_Weight, 
                 subsample = currentSubsampleRate, 
                 colsample_bytree = currentColsampleRate)
  
  xgbcv <- xgb.cv(params = params, 
                  data = Train_xgb1, label = as.numeric(log(y[1:4000])),
                  nrounds = currentNrounds, nfold = 5, 
                  showsd = T, stratified = T, early_stopping_rounds = 20, maximize = F)
  
  testrmse <- xgbcv$evaluation_log$test_rmse_mean[xgbcv$best_iteration]
  trainrmse <- xgbcv$evaluation_log$train_rmse_mean[xgbcv$best_iteration]
  
  return(c(testrmse, trainrmse, currentSubsampleRate, currentColsampleRate,
           currentMin_Child_Weight,currentGamma,currentEta,
           currentMax_Depth,currentNrounds,xgbcv$best_iteration))
  
})

