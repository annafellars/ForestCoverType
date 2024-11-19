library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(themis)
library(DataExplorer)
library(ggplot2)

train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

#EDA
sapply(train_data, class) #all are numeric
plot_histogram(train_data) # some very skewed variables
plot_intro(train_data) #no missing values

correlations <- sapply(train_data, function(x) cor(train_data$Cover_Type, x, use = "complete.obs"))
sorted_correlations <- sort(correlations, decreasing = TRUE, na.last = TRUE)
head(sorted_correlations, 15) #mix of soils and wilderness areas, Id is actually correlated so maybe not random



