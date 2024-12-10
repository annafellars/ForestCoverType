library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(themis)
library(stacks)
library(rules)
library(bonsai)
library(lightgbm)

train_data <- vroom("./train.csv") |>
  mutate(Cover_Type = factor(Cover_Type))

test_data <- vroom("./test.csv")


#Feature Engineering


wild_vars <- paste0("Wilderness_Area", 1:4)
soil_vars <- paste0("Soil_Type", 1:40)



my_recipe <- recipe(Cover_Type~., data = train_data) |>
  step_rm(c("Id", "Soil_Type7", "Soil_Type15")) |>
  step_mutate(total_distance = sqrt(Vertical_Distance_To_Hydrology^2 + 
                                      Horizontal_Distance_To_Hydrology^2)) |>
  step_mutate(Elevation_Plus_Vertical_Hydrology = Elevation + 
                Vertical_Distance_To_Hydrology) |>
  step_mutate(Hydrology_Plus_Fire_Points = Horizontal_Distance_To_Fire_Points + 
                Horizontal_Distance_To_Hydrology) |>
  step_mutate(Hydrology_Plus_Roadways = Horizontal_Distance_To_Roadways + 
                Horizontal_Distance_To_Hydrology) |>
  step_mutate(Fire_Points_Plus_Roadways = Horizontal_Distance_To_Roadways + 
                Horizontal_Distance_To_Fire_Points) |>
  step_normalize(all_numeric_predictors()) |>
  step_zv(all_predictors()) |>
  step_smote(all_outcomes(), neighbors = 4)

prepped_recipe <- prep(my_recipe)
show <- bake(prepped_recipe, new_data = train_data)

#######################################################################################################  
#KNN
knn_model <- nearest_neighbor(neighbors = 100) |>
  set_mode('classification') |>
  set_engine('kknn')

#set workflow
knn_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(knn_model) |>
  fit(data = train_data)

#Predict
knn_preds = predict(knn_wf, 
                    new_data = test_data, type = "class")

knn_kaggle <- knn_preds|>
  bind_cols(test_data) |>
  select(Id, .pred_class) |>
  rename(Cover_Type = .pred_class)

##write out file
vroom_write(x = knn_kaggle, file = "./SimpleKNN.csv", delim=",")


############################################################################################
#boosted trees
library(bonsai)
library(lightgbm)

boost_mod <- boost_tree(tree_depth=tune(),
                        trees=tune(),
                        learn_rate=tune()) |>
  set_engine("lightgbm") |>
  set_mode("classification")

boost_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(boost_mod)

## Set up grid and tuning values
boost_tuning_params <- grid_regular(tree_depth(),
                                    trees(),
                                    learn_rate(),
                                    levels = 5)

##Split data for CV
boost_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
boost_CV_results <- boost_wf |>
  tune_grid(resamples = boost_folds,
            grid = boost_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))
#Find best tuning parameters
boost_best_tune <- boost_CV_results |>
  select_best(metric = "accuracy") #did the same with roc_auc

best_params <- boost_CV_results |>
  select_best(metric = "accuracy")

# Show the best parameters
print(best_params)

##finalize the workflow and fit it
boost_final <- boost_wf |>
  finalize_workflow(boost_best_tune) |>
  fit(data = train_data)

##predict
boost_preds <- boost_final |>
  predict(new_data = test_data, type = "class")

boost_kaggle <- boost_preds|>
  bind_cols(test_data) |>
  select(Id, .pred_class) |>
  rename(Cover_Type = .pred_class)

##write out file
vroom_write(x = boost_kaggle, file = "./SimpleBoost.csv", delim=",")

###################################################################################
#Random Forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) |>
  set_engine("ranger") |>
  set_mode("classification")

## Create a workflow with recipe
forest_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(forest_mod)

## Set up grid and tuning values
forest_tuning_params <- grid_regular(mtry(range = c(1,9)),
                                     min_n(),
                                     levels = 5)

##Split data for CV
forest_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
forest_CV_results <- forest_wf |>
  tune_grid(resamples = forest_folds,
            grid = forest_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))

#Find best tuning parameters
forest_best_tune <- forest_CV_results |>
  select_best(metric = "accuracy")

print(forest_best_tune)

##finalize the workflow and fit it
forest_final <- forest_wf |>
  finalize_workflow(forest_best_tune) |>
  fit(data = train_data)

##predict
forest_preds <- forest_final |>
  predict(new_data = test_data, type = "class")


## Format Predictions for Kaggle
forest_kaggle <- forest_preds|>
  bind_cols(test_data) |>
  select(Id, .pred_class) |>
  rename(Cover_Type = .pred_class)

##write out file
vroom_write(x = forest_kaggle, file = "./SimpleForest.csv", delim=",")

#####################################################################################
##C5 rules
c5_mod <- C5_rules(trees = tune(),
                   min_n = tune()) |>
  set_engine("C5.0") |>
  set_mode("classification") |>
  translate()


## Create a workflow with recipe
c5_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(c5_mod)

## Set up grid and tuning values
c5_tuning_params <- grid_regular(trees(c(1,100)),
                                 min_n(),
                                 levels = 5)

##Split data for CV
c5_folds <- vfold_cv(train_data, v = 10, repeats = 1)

##Run the CV
c5_CV_results <- c5_wf |>
  tune_grid(resamples = c5_folds,
            grid = c5_tuning_params,
            metrics = metric_set(accuracy))

#Find best tuning parameters
c5_best_tune <- c5_CV_results |>
  select_best(metric = "accuracy")
print(c5_best_tune)

##finalize the workflow and fit it
c5_final <- c5_wf |>
  finalize_workflow(c5_best_tune) |>
  fit(data = train_data)

##predict
c5_preds <- c5_final |>
  predict(new_data = test_data, type = "class")


## Format Predictions for Kaggle
c5_kaggle <- c5_preds|>
  bind_cols(test_data) |>
  select(Id, .pred_class) |>
  rename(Cover_Type = .pred_class)

##write out file
vroom_write(x = c5_kaggle, file = "./c5Forest.csv", delim=",")

#####################################################################################
##Stacks
## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Create a control grid
untune_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

##KNN Model
knn_mods <- knn_wf |>
  tune_grid(resamples = folds,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy),
            control = tuned_model)

##Boosted Trees Model
boost_mods <- boost_wf |>
  tune_grid(resamples = folds,
            grid = boost_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy),
            control = untune_model)

##Random Forest Model
forest_mods <- forest_wf |>
  tune_grid(resamples = folds,
            grid = forest_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy),
            control = untune_model)

##Specify with models to include
my_stack <- stacks() |>
  add_candidates(knn_mods) |>
  add_candidates(forest_mods) |>
  add_candidates(boost_mods)

##Fit the stacked model
stack_mod <- my_stack |>
  blend_predictions() |>
  fit_members()

##Use the stacked data to get a prediction
stack_preds <- stack_mod |>
  predict(new_data = test_data)

## Format Predictions for Kaggle
stacked_kaggle <- stack_preds|>
  bind_cols(test_data) |>
  select(Id, .pred_class) |>
  rename(Cover_Type = .pred_class)

##write out file
vroom_write(x = stacked_kaggle, file = "./StackedSimple.csv", delim=",")
