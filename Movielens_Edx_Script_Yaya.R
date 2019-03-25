---
title: "MOVIELENS Project Script"
author: "by Yaya Bamba"
output:
---




## SETTING THE DATASET 
# load the required libraries -
 library(caret, warn.conflicts = FALSE, quietly=TRUE)
 library(data.table, warn.conflicts = FALSE, quietly=TRUE)
 library(tidyverse, warn.conflicts = FALSE, quietly=TRUE)
 library(dslabs, warn.conflicts = FALSE, quietly=TRUE)
 library(dplyr, warn.conflicts = FALSE, quietly=TRUE)
 library(stringr, warn.conflicts = FALSE, quietly=TRUE)
 library(lubridate, warn.conflicts = FALSE, quietly=TRUE)
 library(e1071, warn.conflicts = FALSE, quietly=TRUE)
 library(corrplot, warn.conflicts = FALSE, quietly=TRUE)
 library(ggplot2, warn.conflicts = FALSE, quietly=TRUE)
 library(gtable, warn.conflicts = FALSE, quietly=TRUE)
 library(grid, warn.conflicts = FALSE, quietly=TRUE)
 library(gridExtra, warn.conflicts = FALSE, quietly=TRUE)


# load Movielens dataset -
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                       col.names = c("userId", "movieId", "rating", "timestamp"))

# unzip the dataset and assign  readable column names

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
 colnames(movies) <- c("movieId", "title", "genres")
 movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                            title = as.character(title),
                                            genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data

set.seed(1)
 test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
 edx <- movielens[-test_index,]
 temp <- movielens[test_index,]

 # Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
 edx <- rbind(edx, removed)
 rm(dl, ratings, movies, test_index, temp, movielens, removed)


## DATA PRE-PROCESSING


#extracting the premier year
premierdate <- stringi::stri_extract(edx$title, regex = "(\\d{4})", comments = TRUE ) %>% as.numeric()

#Add the premier year as a column to edx dataset
edx <- edx %>% mutate(premier_year = premierdate)
head(edx)



# convert the rating timestamp to year and add it to edx dataset

edx <- mutate(edx, year_rated = year(as_datetime(timestamp)))


# look at the new edx table to make sure we are good
head(edx)



# check if there is a premier year greater than the year 2019
edx %>% filter(premier_year > 2019) %>% group_by(movieId, title, premier_year) %>% summarize(n = n())

# check is there is a premier date inferior to the year 1900

edx %>% filter(premier_year < 1900) %>% group_by(movieId, title, premier_year) %>% summarize(n = n())



#replace the incorrect premier years after looking up on the movie title

edx[edx$movieId == "671", "premier_year"] <- 1996
edx[edx$movieId == "2308", "premier_year"] <- 1973
edx[edx$movieId == "4159", "premier_year"] <- 2001
edx[edx$movieId == "5310", "premier_year"] <- 1985
edx[edx$movieId == "8864", "premier_year"] <- 2004
edx[edx$movieId == "27266", "premier_year"] <- 2004
edx[edx$movieId == "1422", "premier_year"] <- 1997
edx[edx$movieId == "4311", "premier_year"] <- 1998
edx[edx$movieId == "5472", "premier_year"] <- 1972
edx[edx$movieId == "6290", "premier_year"] <- 2003
edx[edx$movieId == "6645", "premier_year"] <- 1971
edx[edx$movieId == "8198", "premier_year"] <- 1960
edx[edx$movieId == "8905", "premier_year"] <- 1992
edx[edx$movieId == "53953", "premier_year"] <- 2007


## FEATURES ENGINEERING



#Calculate the  movie age  by the time the movie was rated by the user and add the variable to the dataset edx

edx <- edx %>% mutate( movie_age = year_rated - premier_year)

#Calculate average rating by movie add the column to edx dataset

edx <- edx %>% group_by(movieId) %>% mutate(movie_avg_rat = mean(rating))

# Calculate average rating by  user  and add the column to edx dataset

edx <- edx %>% group_by(userId) %>% mutate(user_avg_rat = mean(rating))

# Calculate average rating by movie age and add the column to edx dataset

edx <- edx %>% group_by(movie_age) %>% mutate(age_avg_rat = mean(rating))

# Calculate   the number of  votes per movie and add the column to edx dataset - this this how many time a specific movie was rated

edx <- edx %>% group_by(movieId) %>% mutate(numbVotes = length(unique(userId)))


## DATA PREPARATION


# Use the summary statistic to find the  75th and 25th percentiles
  summary(edx$rating)


# calculate interquartile IQR - this will yield same  75th and 25th percentiles as summary function

    Q1 <- quantile(edx$rating, 0.25)
    Q3 <- quantile(edx$rating, 0.75)
    IQR = Q3 - Q1

# Find the lower and upper fence that defines the outliers

  lowerFence <- Q1 - 1.5 * IQR
  upperFence <- Q3 + 1.5 * IQR
  lowerFence #this is the 25th percentile
  upperFence #this is the 75th percentiel


#define the outliers values from columnn ratings 
OutVals <- boxplot(edx$rating)$out

#print the outliers values
which(edx$rating %in% OutVals)


# count the number of outliers

length(which(edx$rating %in% OutVals))


# Find the outliers 
outliers <- boxplot(edx$rating, plot=FALSE)$out

# First you need find in which rows the outliers are

 edx[which(edx$rating %in% outliers),]

# Now you can remove the rows containing the outliers, one possible option is:

edx_ml <- edx[-which(edx$rating %in% outliers),]



# drop unwanted colums
edx_ml <- edx_ml %>% select(-timestamp, -premier_year, -year_rated)


## MODELLING 

#LETS CACULATE RMSE

## MODEL 1 (movie_effect + user_effect) TRAINING : Predicted_rating =  mu + b_m + b_u

# define RMSE function
RMSE <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}

#Choose the tuning value of lambda

lambdas <- seq(0,5,.5)
model_1_rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_ml$rating)
  
  b_m <- edx_ml %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n() + l))
  
  b_u <- edx_ml %>%
    left_join(b_m, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n() +l))
  
  predicted_rating <- edx_ml %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_m +  b_u) %>% .$pred
  
  return(RMSE(predicted_rating, edx_ml$rating))
})


# lets compute lambdas curve to visually assess the optimal lambda
qplot(lambdas, model_1_rmses)



# determine optimal lambda
lambdas[which.min(model_1_rmses)]


#Check model 1 againt the validation set Prepare Validation set
mu <- mean(validation$rating)
l <- 0.5
b_m <- validation %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n() + l))
  
b_u <- validation %>%
    left_join(b_m, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n() +l))
  
predicted_rating <- validation %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_m +  b_u) %>% .$pred

RMSE(predicted_rating, validation$rating) 


## Model 2 (movie_effect + user_effect  + age_effect ) : predicted_rating =  mu + b_m + b_u + b_a**

# define RMSE2 function
RMSE2 <- function(actual_rating, predicted_rating2){
  sqrt(mean((actual_rating - predicted_rating2)^2))
}

#Choose the tuning value of lambda
lambdas <- seq(0,5,.5)
model_2_rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_ml$rating)
  
  b_m <- edx_ml %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n() + l))
  
  b_u <- edx_ml %>%
    left_join(b_m, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n() +l))
  
   b_a <- edx_ml %>%
    left_join(b_m, by='movieId') %>%
     left_join(b_u, by='userId') %>%
     group_by(movie_age) %>%
    summarize(b_a = sum(rating - b_m - b_u - mu)/(n() +l))
  
  
  
  predicted_rating2 <- edx_ml %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_a, by = "movie_age") %>%
    mutate(pred = mu + b_m +  b_u + b_a) %>% .$pred
  
  return(RMSE2(predicted_rating2, edx_ml$rating))
})


#plot Lambdas
qplot(lambdas, model_2_rmses)


# define optimal lambda value
lambdas[which.min(model_2_rmses)]


# Check model 2 against the validation set : predicted_rating =  mu + b_m + b_u + b_a

#extracting the premier year
premierdatev <- stringi::stri_extract(validation$title, regex = "(\\d{4})", comments = TRUE ) %>% as.numeric()


#Add the premier year as a column to validation dataset
validation2 <- validation %>% mutate(premier_yearv = premierdatev)


# convert the rating timestamp to year and add it to validation dataset

validation2 <- mutate(validation2, year_ratedv = year(as_datetime(timestamp)))


#Calculate the  movie age  by the time the movie was rated by the user and add the variable to the dataset edx

validation2 <- validation2 %>% mutate( movie_age = year_ratedv - premier_yearv)


#Check model 2 againt the validation set Prepare Validation set
mu <- mean(validation2$rating)
l <- 0.5
b_m <- validation2 %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n() + l))
  
b_u <- validation2 %>%
    left_join(b_m, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n() +l))

b_a <- validation2 %>%
    left_join(b_m, by='movieId') %>% 
    left_join(b_u, by='userId') %>% 
    group_by(movie_age) %>%
    summarize(b_a = sum(rating - b_m - b_u - mu)/(n() +l))
  
predicted_rating2 <- validation2 %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_a, by = "movie_age") %>%
    mutate(pred = mu + b_m +  b_u + b_a) %>% .$pred

RMSE2(predicted_rating2, validation2$rating) 


# MODELING WITH OUTLIERS

edx <- edx %>% select(-numbVotes) # drop non numeric values from edx dataset

# define RMSE function for edx
RMSE_all <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}

#Choose the tuning value of lambda

lambdas <- seq(0,5,.5)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  
  b_m <- edx %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n() + l))
  
  b_u <- edx %>%
    left_join(b_m, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n() +l))
  
  predicted_rating <- edx %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_m +  b_u) %>% .$pred
  
  return(RMSE_all(predicted_rating, edx$rating))
})


# lets compute lambdas curve to visually assess the optimal lambda
qplot(lambdas, rmses)

# define optimal lambda value
lambdas[which.min(rmses)]


#Check  againt the validation 
mu <- mean(validation$rating)
l <- 0.5
b_m <- validation %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n() + l))
  
b_u <- validation %>%
    left_join(b_m, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n() +l))
  
predicted_rating <- validation %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_m +  b_u) %>% .$pred

RMSE_all(predicted_rating, validation$rating) 


## PROJECT CONCLUSIONS 

# Computing the 2 models results side by side

#model 1:  Predicted_rating = intercept + movie_effect + user_effect =  mu + b_m + b_u 
movie_User_effect <- RMSE(predicted_rating, validation$rating)
model_1_results <- data_frame(method = "Movie + User_effects", RMSE = movie_User_effect)

# model 2 : Model 2 : predicted_rating = intercept + movie_effect + user_effect  + age_effect=  mu + b_m + b_u + b_a
movie_user_age_effect <- RMSE2(predicted_rating2, validation2$rating)
model_2_results <- data_frame(method = "Movie + User + age_effects", RMSE2 = movie_user_age_effect)


#result table
rmse_results <- bind_rows(model_1_results, model_2_results)
rmse_results

