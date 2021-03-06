##################################################################################
### Step 1: Load Packages

# clean up space
rm(list = ls(all = TRUE))

# load packages
packages <- c('stringr','reshape2','ggplot2','dplyr','pROC','caret','randomForest')
lapply(packages, require, character.only = T)

# clean-up
rm(packages)

##################################################################################
### Step 2: Read Data and Prepare Data                                         ###

# read data
rawdata <- read.csv("../data/churn/churn_fix.csv", header = T)

# check data
str(rawdata)

# create a working copy of the raw data
data <- rawdata

#########################################
# Feature Construction: average mins per call

data$day_avg_mins <- ifelse(data$day_calls == 0, -1, data$day_mins / data$day_calls)
data$eve_avg_mins <- ifelse(data$eve_calls == 0, -1, data$eve_mins / data$eve_calls)
data$night_avg_mins <- ifelse(data$night_calls == 0, -1, data$night_mins / data$night_calls)

str(data)

#########################################
# Correlation Matrix

# creating a copy for transforming columns to numeric
num_data <- data; vars <- colnames(num_data)
catVars <- vars[sapply(num_data[, vars], class) %in% c('factor','character')]
for(v in catVars) {num_data[, v] <- as.numeric(as.factor(num_data[,v]))}

# correlation data
cor_data <- melt(cor(num_data))
names(cor_data) <- c('v1', 'v2', 'correlation')
cor_data <- mutate(cor_data, correlation = round(correlation, 2))

# check correlation data
str(cor_data)
head(cor_data)

# plot correlation
ggplot(cor_data, aes(v1, v2, fill = correlation, label = correlation)) + 
  geom_tile() +
  geom_text(colour = "white") +
  scale_fill_gradient(low="white", high="black") +
  xlab("") + ylab("") + 
  ggtitle("Correlation Matrix")

# clean up
rm(num_data, cor_data, catVars, v, vars)

#########################################
# get state groups

# construct a new column state_group
# in which states with higher frequencies are kept
# and states with lower frequencies are grouped to "other"
data$state <- as.character(data$state)
a <- as.data.frame(table(data$state))
states_select <- as.vector(a[a$Freq > 65, ][[1]])
data$state_group <- ifelse(
  data$state %in% states_select,
  data$state,
  "other"
  )

# change the state columns to factor
state_col <- c("state", "state_group")
for(v in state_col) {data[,v] <- as.factor(data[,v])}

# check result
str(data[, state_col])
head(data[, state_col])

# clean up
rm(a, v, states_select, state_col)

#########################################
# remove un-related and redundant columns

cols_remove <- c(
  "customer_id", "days_renew", "state", 
  "area_code", "phone", "is_vmail_plan",
  "day_charge", "eve_charge", "night_charge"
  )
data <- data[, setdiff(colnames(data), cols_remove)]

# clean up
rm(cols_remove)

str(data)

#########################################

# check missing data
if (nrow(data[!complete.cases(data),]) > 0) {print("There is missing data!!!")}

##################################################################################
### Step 3: random forest                                                      ###

# first, we need to split data into training and testing
splitdf <- function(dataframe, seed = NULL, ratio = 0.5) {
  if (!is.null(seed))
    set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index) * ratio))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset=trainset,testset=testset)
}

# random seeding
getseed = as.integer(runif(1, 1, 100000))
# record the seed for reproducibility
print(paste("Seeding for split:", getseed))

# split data by 2:1 ratio since the data set is small
splits <- splitdf(data, seed = getseed, ratio = 2/3)
training <- splits$trainset
testing <- splits$testset

# check training and testing data
str(training)
str(testing)

# clean up
rm(getseed, splits, splitdf)

#########################################
# Run Random Forest

model <- randomForest(
  is_churn~.
  ,data = training
  ,ntree = 100
  ,mtry = 5 # take square root of number of features
  ,replace = T
  ,importance = T
)

#########################################
# Model evaluation: ROC - Receiver operating characteristic

target = "is_churn"
target_col = grep(target, colnames(training))

# plot ROC curves for training and testing data
training_actual <- training[, target_col]
training_predicted <- predict(
  model, newdata = training[ , - target_col], type = "prob")
testing_actual <- testing[, target_col]
testing_predicted <- predict(
  model, newdata = testing[ , - target_col], type = "prob")

# plot ROC
par(mfrow = c(1,2))
par(pty = "s") # square plotting region
plot.roc(training_actual, training_predicted[, 2], col = "blue", print.auc = T)
plot.roc(testing_actual, testing_predicted[, 2], col = "red", print.auc = T)
par(mfrow = c(1,1))

# clean up
rm(training_actual, training_predicted,
   testing_actual, testing_predicted, target,
   testing, training, target_col)

#########################################
# Model evaluation: Top Features

par(pty = "s") # square plotting region
varImpPlot(
  model
  ,type = 1 # mean decrease in accuracy
  ,n.var = 10
  ,main = "Top Features")