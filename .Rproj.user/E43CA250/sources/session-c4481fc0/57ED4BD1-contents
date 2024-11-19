library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(patchwork)
library(poissonreg)
library(bestglm)
library(rpart)
library(ranger)
library(stacks)
library(dbarts)




### read in data
test_data <- vroom("test.csv")
train_data <- vroom("train.csv")


## EDA
glimpse(train_data)
plot_intro(train_data)
plot_correlation(train_data)
plot_bar(train_data)
plot_histogram(train_data)


## graphs

#barplot of weather
graph1 <- ggplot(train_data, aes(x = factor(weather), y = count, fill = factor(weather))) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("1" = "orange", "2" = "lightblue", "3" = "navy", "4" = "gray"),
                    labels = c("1" = "Sunny", "2" = "Cloudy", "3" = "Stormy", "4" = "Bad Storm")) +
  labs(x = "Weather", y = "Number of Total Rentals")

#scatterplot of count v temp + season
graph2 <- ggplot(train_data, aes(x = temp, y = count, color = factor(season))) +
  geom_point()+
  scale_color_manual(values = c("1" = "pink", "2" = "yellow", "3" = "darkorange", "4" = "lightblue"),
                     labels = c("1" = "Spring", "2" = "Summer", "3" = "Fall", "4" = "Winter")) +
  labs(x = "Tempurature (Celsius)", y = "Number of Total Rentals")

#barplot of working day
graph3 <- ggplot(train_data, aes(x = factor(workingday), y = count, fill = factor(weather))) +
  geom_bar(stat = "identity", position = "dodge") + 
  scale_fill_manual(values = c("1" = "orange", "2" = "lightblue", "3" = "navy", "4" = "gray"),
                    labels = c("1" = "Sunny", "2" = "Cloudy", "3" = "Stormy", "4" = "Bad Storm")) +
  scale_x_discrete(labels = c("0" = "Weekend", "1" = "Workday")) + 
  labs(x = "Day Type", y = "Number of Total Rentals", fill = "Weather")
  

#scatterplot of wind v count
graph4 <- ggplot(train_data, aes(x = windspeed, y = count, color = humidity)) +
  geom_point()+
  labs(x = "Wind Speed", y = "Number of Total Rentals")


(graph1 + graph2) / (graph3 + graph4) 


#####################################################################################
## clean data
train_data <- train_data |>
  select(-casual, -registered) |>
  mutate(count = log(count))
  

cleaning_recipe <- recipe(count~., data = train_data) |>
  step_mutate(weather = ifelse(weather == 4,3,weather)) |>
  step_mutate(weather = factor(weather, labels = c("sunny", "cloudy", "stormy"))) |>
  step_time(datetime, features = "hour") |>
  step_date(datetime, features = "month") |>
  step_date(datetime, features = "year") |>
  step_mutate(hour=factor(datetime_hour)) |>
  step_interact(~ hour:workingday) |>
  step_interact(~ datetime_year:workingday) |>
  step_rm(datetime, datetime_hour) |>
  step_mutate(season = factor(season, labels = c("spring", "summer", "fall", "winter"))) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())
  
  
prepped_recipe <- prep(cleaning_recipe)
show <- bake(prepped_recipe, new_data = train_data)

  
###################################################################################
## Define Linear Regression Model
linmod <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(linmod) |>
  fit(data=train_data)

## Run all the steps on test data
log_preds <- predict(bike_workflow, new_data = test_data)
linear_preds <- log_preds |>
  mutate(.pred = exp(.pred))
linear_preds
     
## Format predictions for Kaggle
kaggle <- linear_preds|>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = kaggle, file = "./BikeSharePreds3.csv", delim=",")


###################################################################################
##Poisson Regression Model 
pois_model <- poisson_reg() |>
  set_engine("glm") |>
  set_mode("regression")

##Combine into a workflow and fit
pois_workflow <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(pois_model) |>
  fit(data = train_data)

##Run all the steps on test data
pois_log_preds <- predict(pois_workflow, new_data = test_data)
pois_preds <- pois_log_preds |>
  mutate(.pred = exp(.pred))

## Format Pois Predictions for Kaggle
pois_kaggle <- pois_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = pois_kaggle, file = "./BikePoisPreds2.csv", delim=",")


###################################################################################
##Penalized regression model
penmod <- linear_reg(penalty = 1, mixture = 0.001) |>
  set_engine("glmnet") |>
  set_mode("regression")

## what I tried 1/0.5 -> 1.28
## 3/0.75 -> 1.41
## 2/0.1 -> 1.18
## 1/0.1 -> 1.09
## 1/0.001 -> 1.06
## 2/0.001 -> 1.11



##Combine into a workflow and fit
penmod_workflow <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(penmod) |>
  fit(data = train_data)

##Run all steps on test data
pen_log_preds <- predict(penmod_workflow, new_data = test_data)
pen_preds <- pen_log_preds |>
  mutate(.pred = exp(.pred))

## Format Penalized Regression Predictions for Kaggle
pen_kaggle <- pen_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = pen_kaggle, file = "./BikePRegPreds.csv", delim=",")


####################################################################################
##Penalized regression model 2
pregmod <- linear_reg(penalty=tune(),
                      mixture=tune()) |>
  set_engine("glmnet")

##set workflow
preg_workflow <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(pregmod)

##Grid of values to tune over
tuning_params <- grid_regular(penalty(),
                              mixture(),
                              levels = 20)

##Split data for CV
folds <- vfold_cv(train_data, v = 20, repeats = 1)

##Run the CV
CV_results <- preg_workflow |>
  tune_grid(resamples=folds,
            grid = tuning_params,
            metrics =metric_set(rmse,mae,rsq))


##Find Best Tuning Parameters
best_tune <- CV_results |>
  select_best(metric = "mae")

##finalize the workflow and fit it
final_wf <- preg_workflow |>
  finalize_workflow(best_tune) |>
  fit(data = train_data)

##predict
tune_preds_log <- final_wf |>
  predict(new_data = test_data)
tune_preds <- tune_preds_log |>
  mutate(.pred = exp(.pred))

## Format Penalized Regression 2 Predictions for Kaggle
tune_kaggle <- tune_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = tune_kaggle, file = "./BikeTuningPreds.csv", delim=",")

###################################################################################
## Regression Trees

treemod <- decision_tree(tree_depth = tune(),
                         cost_complexity = tune(),
                         min_n=tune()) |>
  set_engine("rpart") |>
  set_mode("regression")

## Create a workflow with recipe
tree_wf <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(treemod)

## Set up grid and tuning values
tree_tuning_params <- grid_regular(cost_complexity(),
                              min_n(),
                              tree_depth(),
                              levels = 5)

##Split data for CV
tree_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
tree_CV_results <- tree_wf |>
  tune_grid(resamples=tree_folds,
            grid = tree_tuning_params,
            metrics =metric_set(rmse,mae,rsq))

##Find Best Tuning Parameters
tree_best_tune <- tree_CV_results |>
  select_best(metric = "mae")

##finalize the workflow and fit it
tree_final <- tree_wf |>
  finalize_workflow(tree_best_tune) |>
  fit(data = train_data)

##predict
tree_preds_log <- tree_final |>
  predict(new_data = test_data)
tree_preds <- tree_preds_log |>
  mutate(.pred = exp(.pred))

## Format Penalized Regression 2 Predictions for Kaggle
tree_kaggle <- tree_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = tree_kaggle, file = "./BikeTreePreds.csv", delim=",")

#################################################################################
## random forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) |>
  set_engine("ranger") |>
  set_mode("regression")

## Create a workflow with recipe
forest_wf <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(forest_mod)

## Set up grid and tuning values
forest_tuning_params <- grid_regular(mtry(range = c(1,50)),
                                   min_n(),
                                   levels = 5)

##Split data for CV
forest_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
forest_CV_results <- forest_wf |>
  tune_grid(resamples=forest_folds,
            grid = forest_tuning_params,
            metrics =metric_set(rmse,mae,rsq))

##Find Best Tuning Parameters
forest_best_tune <- forest_CV_results |>
  select_best(metric = "rmse")

##finalize the workflow and fit it
forest_final <- forest_wf |>
  finalize_workflow(forest_best_tune) |>
  fit(data = train_data)

##predict
forest_preds_log <- forest_final |>
  predict(new_data = test_data)
forest_preds <- forest_preds_log |>
  mutate(.pred = exp(.pred))

## Format Penalized Regression 2 Predictions for Kaggle
forest_kaggle <- forest_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = forest_kaggle, file = "./BikeForestPreds.csv", delim=",")


####################################################################################
##Stacks

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Create a control grid
untune_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

###Penalized Regression Model
pregmod <- linear_reg(penalty=tune(),
                      mixture=tune()) |>
  set_engine("glmnet")

##set workflow
preg_workflow <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(pregmod)

##Grid of values to tune over
tuning_params <- grid_regular(penalty(),
                              mixture(),
                              levels = 10)

## Run the CV
preg_mods <- preg_workflow |>
  tune_grid(resamples = folds,
            grid = tuning_params,
            metrics = metric_set(rmse,mae,rsq),
            control = untune_model)

###Random Forest Model
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 250) |>
  set_engine("ranger") |>
  set_mode("regression")

## Create a workflow with recipe
forest_wf <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(forest_mod)

## Set up grid and tuning values
forest_tuning_params <- grid_regular(mtry(range = c(1,50)),
                                     min_n(),
                                     levels = 5)

## Run the CV
forest_mods <- forest_wf |>
  tune_grid(resamples = folds,
            grid = forest_tuning_params,
            metrics = metric_set(rmse,mae,rsq),
            control = untune_model)

##BART Model
bart_mod <- parsnip::bart(mode = "regression",
                          trees = 500)


## Create a workflow with recipe
bart_wf <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(bart_mod) |>
  fit(data = train_data)

##Fit model
tuned_bartmod <- fit_resamples(bart_wf,
              resamples = folds,
              metrics = metric_set(rmse,mae,rsq),
              control = tuned_model)

##Specify with models to include
my_stack <- stacks() |>
  add_candidates(preg_mods) |>
  add_candidates(forest_mods) |>
  add_candidates(tuned_bartmod)

##Fit the stacked model
stack_mod <- my_stack |>
  blend_predictions() |>
  fit_members()

##Use the stacked data to get a prediction

stack_preds_log <- stack_mod |>
  predict(new_data = test_data)
stack_preds <- stack_preds_log |>
  mutate(.pred = exp(.pred))

## Format Penalized Regression 2 Predictions for Kaggle
stack_kaggle <- stack_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = stack_kaggle, file = "./BikeNewStackPreds.csv", delim=",")

############################################################################################
## Bayesian additive regression trees
bart_mod <- parsnip::bart(mode = "regression",
                 trees = 1000)


## Create a workflow with recipe
bart_wf <- workflow() |>
  add_recipe(cleaning_recipe) |>
  add_model(bart_mod) |>
  fit(data = train_data)


##predict
bart_preds_log <- bart_wf |>
  predict(new_data = test_data)
bart_preds <- bart_preds_log |>
  mutate(.pred = exp(.pred))

## Format Penalized Regression 2 Predictions for Kaggle
bart_kaggle <- bart_preds |>
  bind_cols(test_data) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0,count)) |>
  mutate(datetime = as.character(format(datetime)))

##write out file
vroom_write(x = bart_kaggle, file = "./BikeBartPreds.csv", delim=",")
