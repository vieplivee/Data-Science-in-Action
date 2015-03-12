##################################################################################

rm(list = ls(all = TRUE)) # clean up space

# load packages
packages <- c('data.table', 'plyr', 'dplyr', 'reshape2', 'stringr', # general
              'ggplot2', 'lattice', 'lubridate', # plotting and dates
              'pROC', 'ROCR', # ROC curves
              'caret', 'randomForest', 'unbalanced' # modeling
)
lapply(packages, require, character.only = T);
rm(packages) # clean-up

##################################################################################

# reproducible example
set.seed(1)

# path
path <- "/Users/jilitheyoda/Dropbox/Ji_Academic/DataScienct_Talk/Demo_CustomerChurnModel/"
setwd(path)

##################################################################################

rawdata <- read.csv(str_c(path, "churn_fix.csv"), header = T)

#########################################

data <- rawdata

#########################################

# correlation matrix
num_data <- data
vars <- colnames(data)
catVars <- vars[sapply(data[, vars], class) %in% c('factor','character')]
for(v in catVars) {
  num_data[,v] <- as.numeric(as.factor(num_data[,v]))
}

cor_data <- melt(cor(num_data))
names(cor_data) <- c('v1', 'v2', 'correlation')
cor_data <- mutate(cor_data, correlation = round(correlation, 2))
ggplot(cor_data, aes(v1, v2, fill = correlation, label = correlation)) + 
  geom_tile() +
  geom_text(colour = "white") +
  scale_fill_gradient(low="white", high="black") +
  xlab("") + ylab("") + 
  ggtitle("Correlation Matrix")

#########################################

# get state groups
data$state <- as.character(data$state)
a <- as.data.frame(table(data$state))
states_select <- as.vector(a[a$Freq > 65, ][[1]])
data$state_group <- ifelse(data$state %in% states_select, data$state, "other")

#########################################

# remove un-related columns
cols_remove <- c(
  "customer_id", "days_renew", "state", 
  "area_code", "phone", "is_vmail_plan",
  "day_charge", "eve_charge", "night_charge")
data <- data[, setdiff(colnames(data), cols_remove)]

#########################################

# get average mins per call
data$day_avg_mins <- ifelse(data$day_calls == 0, -1, data$day_mins / data$day_calls)
data$eve_avg_mins <- ifelse(data$eve_calls == 0, -1, data$eve_mins / data$eve_calls)
data$night_avg_mins <- ifelse(data$night_calls == 0, -1, data$night_mins / data$night_calls)

#########################################

# convert character to factor
vars <- colnames(data)
catVars <- vars[sapply(data[, vars], class) %in% c('factor','character')]
for(v in catVars) {
  data[,v] <- as.factor(data[,v])
}

#########################################

# check missing ata
data[!complete.cases(data),]

##################################################################################

# split data

splitdf <- function(dataframe, seed = NULL, ratio = 0.5) {
  if (!is.null(seed))
    set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index) * ratio))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset=trainset,testset=testset)
}
getseed = as.integer(runif(1, 1, 100000)); print(paste("Seeding for split:", getseed))
splits <- splitdf(data, seed = getseed, ratio = 1/2)
lapply(splits, nrow)
lapply(splits, head)
training <- splits$trainset
testing <- splits$testset

#########################################

# plant a random forest
model <- randomForest(
  is_churn~.
  ,data = training
  ,ntree = 100
  ,mtry = 5
  ,replace = T
  ,importance = T
  ,do.trace = T
)

varImp(model)
varImpPlot(model, type=1, n.var = 10, main = "Random Forest")

#########################################

target = "is_churn"

# plot ROC curves for training and testing data
training_actual <- training[, grep(target, colnames(training))]
training_predicted <- predict(
  model, newdata = training[ , - grep(target, colnames(training))], type = "prob")
testing_actual <- testing[, grep(target, colnames(testing))]
testing_predicted <- predict(
  model, newdata = testing[ , - grep(target, colnames(testing))], type = "prob")
par(mfrow = c(1,2))
plot.roc(training_actual, training_predicted[, 2], col = "blue", print.auc = T)
plot.roc(testing_actual, testing_predicted[, 2], col = "red", print.auc = T)
par(mfrow = c(1,1))

#########################################

# optimize the cutoff for testing data
pred <- prediction(testing_predicted[, 2], testing_actual)
perf <- performance(pred, measure="acc", x.measure="cutoff")
bestAccInd <- which.max(perf@"y.values"[[1]])
bestMsg <- print(paste("best accuracy = ", perf@"y.values"[[1]][bestAccInd],
                       " at cutoff = ", round(perf@"x.values"[[1]][bestAccInd],4), sep=""))
plot(perf, type = "b", sub = bestMsg)