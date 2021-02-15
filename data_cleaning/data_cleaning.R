library(jsonlite)
library(dplyr)
library(sf)
library(ggplot2)

setwd("C:/Users/SL-webserver-admin/Desktop/Haydn_YelpProject_May2020")

## Read in all the businesses from the YELP JSON file and create a spatial dataframe
business.infile <- stream_in(file("data/yelp_dataset/yelp_academic_dataset_business.json"),pagesize = 100000)
business.infile <- select(business.infile, business_id, name, categories, latitude, longitude, stars, review_count)
business.pts <- sf::st_as_sf(business.infile, coords = c("longitude","latitude"), crs = 4269)

## Get the Toronto CMA shapefile and fine all of the businesses within it
CMAs <- read_sf(dsn = "data/CMAs", layer = "lcma000b16a_e")
toronto.cma <- CMAs[CMAs$CMANAME == "Toronto",]
toronto.cma <- st_transform(toronto.cma, crs=4269)  
toronto.pts <- st_join(business.pts, toronto.cma, join = st_intersects)
toronto.pts <- toronto.pts[which(toronto.pts$PRUID=="35"),]
## This is just to make variables easier to read later (business, reviews, users are the main dataframes)
business <- toronto.pts
business <- select(business, business_id, categories, geometry)
#st_write(business, "data\\yelp_dataset\\business.shp")
#business <- st_read(""data\\yelp_dataset\\business.shp", package="sf")

## Create a list of businesses (business_id) within the boundaries of the Toronto CMA
#toronto.business.list <- as.list(business$business_id)
## Subset only the reviews that are linked to businesses in Toronto
#reviews.all <- jsonlite::stream_in(file("data/yelp_dataset/yelp_academic_dataset_review.json"), pagesize = 100000)
#reviews.all <- select(reviews.all, review_id, user_id, business_id, stars, text, date)
#reviews <- reviews.all[reviews.all$business_id %in% toronto.business.list,]
#write.csv(reviews, "data\\yelp_dataset\\toronto_reviews.csv", row.names=FALSE)
reviews <- read.csv("data\\yelp_dataset\\toronto_reviews.csv")

## Find all the users that have made at least 1 review for a business in Toronto
#toronto.users.list <- as.list(reviews$user_id)
## Subset only the users that created at least one review in Toronto
#users.tmp <- jsonlite::stream_in(file("data/yelp_dataset/yelp_academic_dataset_user.json"), pagesize = 100000)
#users.tmp <- select(users.tmp, user_id, review_count, elite, average_stars)
#users <- users.tmp[users.tmp$user_id %in% toronto.users.list,]
#write.csv(users, "data\\yelp_dataset\\toronto_users.csv", row.names=FALSE)
users <- read.csv("data\\yelp_dataset\\toronto_users.csv")

## Get how many reviews each user created in Toronto and merge to users df
#toronto.review.counts <- aggregate(reviews$business_id, by = list(reviews$user_id), FUN = length)  
#all.review.counts <- aggregate(reviews.all$business_id, by = list(reviews.all$user_id), FUN = length)
#toronto.review.counts <- rename(toronto.review.counts, tor.counts=x, user_id=Group.1) 
#all.review.counts <- rename(all.review.counts, all.counts=x, user_id=Group.1) 
#write.csv(toronto.review.counts, "data\\yelp_dataset\\toronto_review_counts.csv", row.names=FALSE)
toronto.review.counts <- read.csv("data\\yelp_dataset\\toronto_review_counts.csv")
#write.csv(all.review.counts, "data\\yelp_dataset\\all_review_counts.csv", row.names=FALSE)
all.review.counts <- read.csv("data\\yelp_dataset\\all_review_counts.csv")

users <- merge(users, toronto.review.counts, by = "user_id")
users <- merge(users, all.review.counts, by = "user_id")

#####-------------------------------------#####
#users <- users[users$all.counts > 4,]
#####-------------------------------------#####

## If the number of reviews in Toronto is greater than 66.6% of their total reviews, they will be labeled as local
users$local <- 0
## The variable for review counts is n after the merge
users$local[users$tor.counts / users$all.counts >= 0.666] <- 1

## If the review was made in a year the user was elite, set isElite to 1
reviews <- dplyr::left_join(reviews, users, by = "user_id")
reviews$year <- substr(reviews$date, start = 1, stop = 4)
reviews$isElite <- 0
reviews$isElite[grepl(reviews$year, reviews$elite)] <- 1

## Add categories of the business to the reviews
reviews <- dplyr::left_join(reviews, business.infile, by = "business_id")

## Change the reviews into a spatial data frame
reviews.pts <- sf::st_as_sf(reviews, coords = c("longitude","latitude"), crs = 4269)
reviews.pts <- rename(reviews.pts, 
                      user.average.stars=average_stars, 
                      user.elite.years=elite,
                      user.total.reviews=all.counts,
                      user.total.toronto.reviews=tor.counts, 
                      user.complete.reviews.donotuse=review_count.x,
                      business.star.value=stars.y, 
                      business.total.reviews=review_count.y,
                      business.name=name,
                      business.categories=categories,
                      review.isElite=isElite,
                      review.isLocal=local,
                      review.year=year,
                      review.text=text,
                      review.date=date,
                      review.star.value=stars.x
)

#write.csv(reviews.pts, "data\\yelp_dataset\\completed_reviews.csv", row.names=FALSE)
#reviews.pts <- read.csv("data\\yelp_dataset\\completed_reviews.csv")

# rm(reviews.tmp, users.tmp, toronto.business.list, toronto.users.list, toronto.review.counts)
# rm(CMAs, toronto.pts, business.pts, business.infile)
