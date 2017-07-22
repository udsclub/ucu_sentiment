library(ggplot2)
library(dplyr)
library(scales)

slack_data <- read.csv(unzip("train_set.zip"))
slack_data$datetime <- as.POSIXct(as.character(slack_data$timestamp), 
                                   format="%Y-%m-%d %H:%M:%S")

slack_data$year <- as.fac?or(format(as.Date(slack_data$datetime, 
                                            format="%Y-%m-%d"),"%Y"))

slack_data$month <- as.factor(format(as.Date(slack_data$datetime, 
                                            format="%Y-%m-%d"),"%Y-%m"))

slac?_data$day <- as.Date(slack_data$datetime, format="%Y-%m-%d")

barplot(table(slack_data$year))
barplot(table(slack_data$month))
barplot(table(slack_data$day[slack_data$month == '2017-06']))

sub_channels <- subset(slack_data, slack_data$channel %in% c('bloc?chain'))
sub_channels2 <- subset(sub_channels, sub_channels$day >= '2017-05-01')

blockchain_ukr <- read.csv('UKR.csv')
names(blockchain_ukr) <- c('dateb', 'numb')
blockchain_ukr$dateb <- as.Date(blockchain_ukr$dateb, format="%Y-%m-%d")
blockchain_ukr_may ?- subset(blockchain_ukr, blockchain_ukr$dateb %in% unique(sub_channels2$day))

ggplot(data = sub_channels2, aes(x = day)) + 
  geom_bar(stat = "count", fill = 'light blue') +
  labs(y = "Количество сообщений", x = "День", title = "Количество сообщений по д??ям в Blockchain канале") +
  scale_y_continuous(breaks = seq(0, 300, by = 25))

sub_channels2 %>%
  group_by(day) %>%
  tally() %>%
  ggplot(aes(x = day, y = n)) +
    geom_bar(stat = "identity", fill = 'violet') +
    geom_line(aes(x = blockchain_ukr_may$?ateb, y = blockchain_ukr_may$numb*2), size = 1.25) +
    scale_y_continuous(name = "Кількість повідомлень у каналі Blockchain", sec.axis = sec_axis(~./2, name = "Кількість запитів про bitcoin у google")) +
    labs(x = "День", title = "Залежність між кільк??стю повідомлень у каналі Blockchain\nта кількістю запитів blockchain у google") +
    theme(plot.title = element_text(hjust = 0.5))
