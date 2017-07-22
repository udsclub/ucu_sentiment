slack_data <- read.csv(unzip("train_set.zip"))
slack_data$datetime <- as.POSIXct(as.character(slack_data$timestamp), 
                                   format="%Y-%m-%d %H:%M:%S")

slack_data$year <- as.factor(format(as.Date(slack_data$datetime, 
                                            format="%Y-%m-%d"),"%Y"))

slack_data$month <- as.factor(format(as.Date(slack_data$datetime, 
                                            format="%Y-%m-%d"),"%Y-%m"))

slack_data$day <- as.Date(slack_data$datetime, format="%Y-%m-%d")

barplot(table(slack_data$year))
barplot(table(slack_data$month))
barplot(table(slack_data$day[slack_data$month == '2017-06']))
