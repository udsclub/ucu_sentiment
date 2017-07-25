library(ggplot2)
library(plyr)
library(dplyr)
library(reshape2)
library(RColorBrewer)
library(data.table)
library(ggalt)
library(ggstance)

# Loading the data and preparing it for the visualization
# To make all of this stuff work correctly, put all the da?a needed
# for plotting directly into your working directory

slack_data <- read.csv(unzip("train_set.zip"))
slack_data$datetime <- as.POSIXct(as.character(slack_data$timestamp), 
                                   format="%Y-%m-%d %H:%M:%S")

slack_data$y?ar <- as.factor(format(as.Date(slack_data$datetime, 
                                            format="%Y-%m-%d"),"%Y"))

slack_data$month <- as.factor(format(as.Date(slack_data$datetime, 
                                            format="%Y-%m-%d"),"%?-%m"))

slack_data$day <- as.Date(slack_data$datetime, format="%Y-%m-%d")

slack_data$hm <- format(slack_data$datetime, "%H:%M")

slack_data$h <- format(slack_data$datetime, "%H")

# First barplots to have a look at the data (user activity), base R graphic?

barplot(table(slack_data$year))
barplot(table(slack_data$month))
barplot(table(slack_data$day[slack_data$month == '2017-06']))

# Blockchain channel data subsetting

sub_channels <- subset(slack_data, slack_data$channel %in% c('blockchain'))
sub_channels? <- subset(sub_channels, sub_channels$day >= '2017-05-01')

# Loading and preparing data from google trends about Blockchain

blockchain_ukr <- read.csv('UKR.csv')
names(blockchain_ukr) <- c('dateb', 'numb')
blockchain_ukr$dateb <- as.Date(blockchain_ukr$d?teb, format="%Y-%m-%d")
blockchain_ukr_may <- subset(blockchain_ukr, blockchain_ukr$dateb %in% unique(sub_channels2$day))

# The graph for Blockchain data only, ggplot

ggplot(data = sub_channels2, aes(x = day)) + 
  geom_bar(stat = "count", fill = 'light ?lue') +
  labs(y = "Number of messages", x = "Day", title = "Number of messages per day in Blockchain channel") +
  scale_y_continuous(breaks = seq(0, 300, by = 25))

# The graph for both Blockchain channel data and google trends, ggplot

sub_channels2 %>%?  group_by(day) %>%
  tally() %>%
  ggplot(aes(x = day, y = n)) +
    geom_bar(stat = "identity", fill = 'violet') +
    geom_line(aes(x = blockchain_ukr_may$dateb, y = blockchain_ukr_may$numb*2), size = 1.25) +
    scale_y_continuous(name = "Number of mes?ages in Blockchain channel", sec.axis = sec_axis(~./2, name = "Google trend score for requests about bitcoin")) +
    labs(x = "Day", title = "Dependency between the number of messages in Blockchain channel\nand google trend score for requests about bitcoi?") +
    theme(plot.title = element_text(hjust = 0.5))

# Preparing the data for users' activity visualization

sub_channels_per_day <- sub_channels %>% group_by(day) %>% tally()
mes_per_day <- slack_data %>% group_by(channel, day) %>% tally()
mes_per_day$?ear <- strftime(mes_per_day$day, format = "%Y")
mes_per_day$month <- months(mes_per_day$day)
mes_per_day$weekday <- factor(weekdays(mes_per_day$day), levels= c("Monday", 
                                                                "Tuesday", "Wednesday?, "Thursday", "Friday", "Saturday","Sunday"))
mes_per_day$week <- as.numeric(format(mes_per_day$day,"%W"))
mes_per_day<-ddply(mes_per_day,.(factor(month)),transform,monthweek=1+week-min(week))
mes_per_day_2016 <- subset(mes_per_day, mes_per_day$year == 201?)
mes_per_day_2017 <- subset(mes_per_day, mes_per_day$year == 2017)

sub_channels_per_day$year <- strftime(sub_channels_per_day$day, format = "%Y")
sub_channels_per_day$month <- months(sub_channels_per_day$day)
sub_channels_per_day$weekday <- factor(weekda?s(sub_channels_per_day$day), levels= c("Monday", 
                                                                   "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"))
sub_channels_per_day$week <- as.numeric(format(sub_channels_per_day$day?"%W"))
sub_channels_per_day<-ddply(sub_channels_per_day,.(factor(month)),transform,monthweek=1+week-min(week))
sub_channels_day_2017 <- subset(sub_channels_per_day, sub_channels_per_day$year == 2017)

# Plotting the heatmap for Thoery and Practice channel ?ctivity (weekdays), ggplot

ggplot(data = na.omit(subset(mes_per_day_2017, mes_per_day_2017$channel == "theory_and_practice")), aes(x = monthweek,
                                             y = weekday, fill = n)) +
  geom_tile(color="blue") + 
  facet_g?id(year~factor(as.yearmon(day))) + 
  scale_fill_gradient(low="light blue", high="dark blue", guide = "colourbar") +
  labs(title = "Number of messages in Theory and Practice channel per day", fill = "Number of messages") +
  theme(plot.title = element_tex?(hjust = 0.5), axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  xlab("Week of Month") + ylab("")

# The base for hourly user activity heatmap (sorting for hour ticks on y axis missing)

subset(slack_data, slack_data$day >= '2017-07-01?) %>%
  group_by(day, h) %>%
  tally() %>%
  ggplot(aes(x = day, y = h, fill = n)) +
  geom_tile(color = "blue") +
  scale_y_discrete(labels=paste(order(unique(slack_data$h)), "00", sep = ":")) +
  scale_fill_gradient(low="light blue", high="dark blue", gu?de = "colourbar") +
  ylab("")

# Loading and preparing data on users asking questions

new_users_qs_2017 <- read.csv("df.csv", encoding = 'UTF-8')
new_users_qs_2017_top_20 <- head(arrange(new_users_qs_2017,desc(n)), n = 20)
new_users_qs_2016 <- na.omit(re?d.csv("df2016", encoding = 'UTF-8'))
new_users_qs_2016_top_20 <- head(arrange(new_users_qs_2016,desc(n)), n = 20)
new_users_qs_2016_top_20$year <- rep(2016,nrow(new_users_qs_2016_top_20))
new_users_qs_2017_top_20$year <- rep(2017,nrow(new_users_qs_2017_top?20))
new_users_qs_top_20 <- rbind(new_users_qs_2016_top_20, new_users_qs_2017_top_20)
new_users_qs_top_20$year <- as.factor(new_users_qs_top_20$year)

# Users asking questions by user, channel, year and number of questions, ggplot

ggplot(data = new_users_?s_top_20, aes(x = reorder(real_name, -n), y = n)) +
  geom_bar(aes(fill = year, width = .6), position = "dodge", stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(hjust = 0.5)) +
  xlab("User name") +
 ?ylab("Number of messages") +
  labs(title = "Number of questions by users per year") +
  scale_fill_brewer(palette = "Set2")

# Loading the data about experts

overall_experts <- read.csv("overall_experts.csv", encoding = 'UTF-8')

# Experts by user, chann?l and number of reactions, ggplot

ggplot(data = subset(overall_experts, overall_experts$reactions_count>15), aes(x = reorder(real_name, -reactions_count), y = reactions_count, width = .6)) +
  geom_bar(aes(fill = channel), stat="identity", position = "dod?e") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(hjust = 0.5)) +
  xlab("User name") +
  ylab("Number of reactions") +
  labs(title = "Top expert users by channels") +
  scale_fill_brewer(palette = "Set2")

# Loadi?g the data about trolls

overall_trolls <- read.csv("overall_trolls.csv", encoding = 'UTF-8')

# Trolls by user, channel and number of reactions, ggplot

overall_trolls %>%
  select(channel, real_name, reactions_count) %>%
  filter(reactions_count>5) %>%
 ?ggplot(aes(x = reorder(real_name, -reactions_count), y = reactions_count, width = .6)) +
  geom_bar(aes(fill = channel), stat="identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(hjust = 0.5?) +
  xlab("User name") +
  ylab("Number of reactions") +
  labs(title = "Top troll users by channels") +
  scale_fill_brewer(palette = "Set2")

# Some stats: how many experts are also trolls

overall_experts %>%
  filter(real_name %in% overall_trolls$real?name) %>%
  filter(reactions_count > 5) %>%
  group_by(real_name) %>%
  tally()

# Some stats and preparation for plotting: how many trolls are also experts

overall_trolls %>%
  filter(real_name %in% overall_experts$real_name) %>%
  filter(reactions_count?> 5) %>%
  select(real_name, reactions_count, channel) %>%
  mutate(gr = -1) -> experts_in_trolls

# Preparing the data frame for visualization on trolls who are also experts

overall_experts %>%
  filter(real_name %in% experts_in_trolls$real_name) %>%
  s?lect(real_name, reactions_count, channel) %>%
  mutate(gr = 1) %>%
  rbind(experts_in_trolls) -> experts_in_trolls_all

# Trolls who are also experts by user, channel and number of reactions, ggplot

ggplot(data = subset(experts_in_trolls_all, experts_in_t?olls_all$reactions_count > 6), aes(x = real_name, y = reactions_count*gr, width = .6)) +
  geom_bar(aes(fill = channel), stat="identity", position = "dodge") +
  coord_flip() +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("User name") +
  ylab(?Number of reactions (negative for trolls, positive for experts)") +
  labs(title = "Troll users who are also experts") +
  scale_fill_brewer(palette = "Set2")