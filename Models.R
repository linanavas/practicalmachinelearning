library(dplyr)
library(caret)
library(ggplot2)

# Loading data

plm_train <- read.csv('pml-training.csv')
plm_test <- read.csv('pml-testing.csv')

# Cleaning data

classe <- plm_train %>% 
  select(X, classe)

plm_train <- plm_train %>% 
  select(-classe) %>% 
  mutate_if(is.factor, 
            function(x)as.numeric(as.character(x))) %>% 
  select_if(~!any(is.na(.))) %>% 
  left_join(classe) %>% 
  select(-raw_timestamp_part_1, 
         -raw_timestamp_part_2,
         -X,
         -num_window)
  
# Spliting data

set.seed(1234)

inTrain <- createDataPartition(plm_train %>% 
                                 pull(classe), 
                               p = 0.6, 
                               list = F)

training <- plm_train[inTrain,]
noTraining <- plm_train[-inTrain,]

inTest <- createDataPartition(noTraining %>% 
                                 pull(classe), 
                               p = 0.5, 
                               list = F)

testing <- noTraining[inTest,]
validation <- noTraining[-inTest,]

# Training models

set.seed(5678)

ctrl <- trainControl(method = 'cv', number = 5)

model_rf <- train(classe ~ ., 
                  method = 'rf',
                  data = training,
                  trControl = ctrl)

model_gbm <- train(classe ~ ., 
                  method = 'gbm',
                  data = training,
                  trControl = ctrl)

# Evaluating models

pred_rf <- predict(model_rf, testing)
pred_gbm <- predict(model_gbm, testing)

matrix_rf <- confusionMatrix(testing$classe, pred_rf)
matrix_gbm <- confusionMatrix(testing$classe, pred_gbm)

tab_rf <- matrix_rf$table
acc_rf <- matrix_rf$overall[1]

tab_gbm <- matrix_gbm$table
acc_gbm <- matrix_gbm$overall[1]

imp_rf <- varImp(model_rf)$importance
imp_rf <- imp_rf %>% 
  mutate(Variable = rownames(.)) %>% 
  arrange(desc(Overall)) %>% 
  filter(row_number()<=10)

fig <- ggplot(aes(y = Variable, x = Overall), data = imp_rf %>% 
         mutate(Variable = factor(Variable, levels = imp_rf %>% 
                                    arrange(Overall) %>% 
                                    pull(Variable)))) + 
  geom_bar(stat = 'identity', fill = 'cornflowerblue') +
  xlab('Importance') + 
  ylab('Feature') + 
  theme_classic()

# Out of sample error

pred_out <- predict(model_rf, validation)
matrix_out <- confusionMatrix(validation$classe, pred_out)
tab_out <- matrix_out$table
acc_out <- matrix_out$overall[1]

save(list = c('tab_rf','acc_rf',
              'tab_gbm','acc_gbm',
              'tab_out','acc_out',
              'fig'), 
              file = 'Results.RData')
