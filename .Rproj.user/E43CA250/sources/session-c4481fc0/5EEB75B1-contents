library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(themis)

missing_data <- vroom("./trainWithMissingValues.csv")
missing_data <- missing_data |>
  mutate(color = as.factor(color))

train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

missing_recipe <- recipe(type~., data = missing_data) |>
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul, color), neighbors = 3) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, bone_length), neighbors = 3) |>
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color, bone_length, rotting_flesh), neighbors = 3)

prepped_recipe <- prep(missing_recipe)
imputed_data <- bake(prepped_recipe, new_data = missing_data)


rmse_vec(train_data[is.na(missing_data)],
         imputed_data[is.na(missing_data)])

#0.1473116

my_recipe <- recipe(type~., data = train_data) |>
  step_mutate_at(color, fn = factor) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) |>
  step_normalize(all_nominal_predictors()) |>
  step_smote(all_outcomes(), neighbors = 4)

prepped_recipe <- prep(my_recipe)
show <- bake(prepped_recipe, new_data = train_data)


######################################################################################
#KNN
knn_model <- nearest_neighbor(neighbors = 20) |>
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

## Format predictions for Kaggle
kaggle <- knn_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGKNN.csv", delim=",")

#####################################################################################
#NB
nb_mod <- naive_Bayes(Laplace= tune(), smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nb_mod)

## Set up grid and tuning values
nb_tuning_params <- grid_regular(Laplace(),
                                 smoothness(),
                                 levels = 5)

##Split data for CV
nb_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
nb_CV_results <- nb_wf |>
  tune_grid(resamples = nb_folds,
            grid = nb_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))
#Find best tuning parameters
nb_best_tune <- nb_CV_results |>
  select_best(metric = "roc_auc")

##finalize the workflow and fit it
nb_final <- nb_wf |>
  finalize_workflow(nb_best_tune) |>
  fit(data = train_data)

##predict
nb_preds <- nb_final |>
  predict(new_data = test_data, type = "class")

kaggle <- nb_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGNB.csv", delim=",")

#################################################################################
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
  select_best(metric = "roc_auc")

##finalize the workflow and fit it
boost_final <- boost_wf |>
  finalize_workflow(boost_best_tune) |>
  fit(data = train_data)

##predict
boost_preds <- boost_final |>
  predict(new_data = test_data, type = "class")

kaggle <- boost_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGboost.csv", delim=",")

########################################################################################
#Bart model

bart_mod <- bart(trees=tune()) |>
  set_engine("dbarts") |>
  set_mode("classification")

bart_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(bart_mod)

## Set up grid and tuning values
bart_tuning_params <- grid_regular(trees(),
                                   levels = 5)

##Split data for CV
bart_folds <- vfold_cv(train_data, v = 5, repeats = 1)

##Run the CV
bart_CV_results <- bart_wf |>
  tune_grid(resamples = bart_folds,
            grid = bart_tuning_params,
            metrics = metric_set(f_meas, sens, recall, 
                                 accuracy))
#Find best tuning parameters
bart_best_tune <- bart_CV_results |>
  select_best(metric = "accuracy")

##finalize the workflow and fit it
bart_final <- bart_wf |>
  finalize_workflow(bart_best_tune) |>
  fit(data = train_data)

##predict
bart_preds <- bart_final |>
  predict(new_data = test_data, type = "class")

kaggle <- bart_preds|>
  bind_cols(test_data) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

##write out file
vroom_write(x = kaggle, file = "./GGGbart.csv", delim=",")
