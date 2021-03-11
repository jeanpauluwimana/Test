# Final Poster
# Author: Jean Paul Uwimana
# IST-719
# Date: March 21, 2020
###########################

# loading required packages
library(dplyr)
library(ggplot2)
library(plotly)

# setting the file path location
path <- paste0("C:\\Users\\Jpuwi\\Documents\\Syracuse_University\\",
              "Winter2020\\IST-719\\Assignments\\WIP\\",
              "acs2017_census_tract_data.csv")

# reading in the file
census <- read.csv(file = path, stringsAsFactors = F, header = T, sep = ",")

# looking at the data
nrow(census)
census[1:4, 2:6]


# data subset to extract columns needed for analysis
census.subset <- select(census, c(State, County, TotalPop, Hispanic, White,
                                  Black, Native, Asian, Pacific, Income,
                                  Poverty, Unemployment))

# getting rid of data with NAs
census.subset <- census.subset[complete.cases(census.subset), ]

##### Adding state abbreviations to our dataframe for easy plotting ############
################################################################################
# Removing DC & PR
census.subset <- dplyr::filter(census.subset, !State %in% c("District of Columbia", "Puerto Rico"))

# binding the state name and state abbraviation using R built-in datasets
state.abbr <- as.data.frame(cbind(state.abb, state.name), stringsAsFactors = F)

# looking up state abbreviation and match it with our state name
census.subset <- left_join(census.subset, state.abbr, by = c("State" = "state.name"))

# re-ordering column names
census.subset <- dplyr::select(census.subset, c(State, state.abb, County, TotalPop, Hispanic,
                                                White, Black, Native, Asian, Pacific, Income,
                                                Poverty, Unemployment))

# function to calculate diversity score
# Start of function 
#_____________________________________________________________________________
diversify <- function(dataframe)
{
  for(i in nrow(dataframe))
  {
    # Diversity calulcation
    # D = 1 - SumOf n(n-1) / N(N-1)
    # n = number of individuals of each species
    # N = total number of individuals of all species
    # Source: http://bit.ly/2vbUEeL
    ################################################
    N <- dataframe$TotalPop * (dataframe$TotalPop - 1)
    Hispanic <- (dataframe$TotalPop * dataframe$Hispanic) / 100
    White <- (dataframe$TotalPop * dataframe$White) / 100
    Black <- (dataframe$TotalPop * dataframe$Black) / 100
    Native <- (dataframe$TotalPop * dataframe$Native) / 100
    Asian <- (dataframe$TotalPop * dataframe$Asian) / 100
    Pacific <- (dataframe$TotalPop * dataframe$Pacific) / 100
    # Other <- (dataframe$Other * dataframe$Other) / 100
    nMinusOne <- (Hispanic * (Hispanic - 1)) + (White * (White - 1)) + 
      (Black * (Black - 1)) + (Native * (Native - 1)) + 
      (Asian * (Asian - 1)) + (Pacific * (Pacific - 1))  # + (Other * (Other - 1))
    
    Diversity.Score = 1 - (nMinusOne / N)
    return(Diversity.Score)
  }
}
#______________________________________________________________________________
# End of funtion

# Summarizing the data by mean income and unemployment rate                   
census.subset %>%
  group_by(State, state.abb) %>%
  dplyr::summarise(Unemployment.Rate = mean(Unemployment), Income = mean(Income)) %>%
  {. ->> b} # saving the result of piping

# converting the summarization results to dataframe 
states <- as.data.frame(b)
# sorting average income in descending order
sorted.states <- states[base::order(-states$Income), ]

# top 10 states by income and their unemployment rate: Question 1 
top.ten <- head(sorted.states, 10)

# Plotting the top 10
plot(top.ten$Unemployment.Rate, top.ten$Income, type = "p", col = "orange", pch = 20,
     main = "Top 10 states with highest income vs Unemployment Rate", 
     xlab = "Unemployment Rate", ylab = "Per Capita Income", family = "mono")
text(top.ten$Unemployment.Rate, top.ten$Income, labels = top.ten$state.abb, pos = 2)


# Top 10 poorest states by poverty level and their per capita income
####################################################################
census.subset %>%
  group_by(State, state.abb) %>%
  dplyr::summarise(Poverty = mean(Poverty), Income = mean(Income)) %>%
  {. ->> x} # saving the result of piping

# converting the summarization results to dataframe 
poverty <- as.data.frame(x)    
# sorting average poverty in descending order
sorted.poverty <- poverty[base::order(-poverty$Poverty), ]

# top 10 states by income and their poverty rate: Question 3 
top.ten1 <- head(sorted.poverty, 10)

# plotting the bottom 10 states 
plot(x = top.ten1$Poverty, y = top.ten1$Income, col = "orange", pch = 20,
     xlab = "Poverty Level", ylab = "Per Capita Income", 
     main = "Income & Poverty relationship for \nthe states with the lowest income",
     family = "mono")
text(top.ten1$Poverty, top.ten1$Income, labels = top.ten1$state.abb, pos = 2)

# Distribution of income
d <- density(census.subset$Income)
plot(d, type = "l", main = "Income Distribution",
     xlab = "Income", family = "mono")
polygon(d, col = "orange", border = F)

# US poverty rate by state
census.subset %>%
  group_by(State) %>%
  dplyr::summarise(Poverty = mean(Poverty)) %>%
  {. ->> y} # saving the result of piping

# converting the summarization results to dataframe 
poverty.rate <- as.data.frame(y)  

### US poverty rate by state
poverty.rate <- tapply(census.subset$Poverty, list(census.subset$State), mean)
# sorting poverty rate in decreasing order
poverty.rate <- sort(poverty.rate, decreasing = T)

# US Poverty Rate map
#####################
library(usmap)
library(ggplot2)
poverty.rate.df = as.data.frame.table(round(poverty.rate, 2))
colnames(poverty.rate.df) <- c("state", "poverty_rate")

plot_usmap(data = poverty.rate.df, values = "poverty_rate",
           color = "white", labels = T) +
  scale_fill_continuous(low = "white", high = "#2F59C5", name = "Poverty Rate",
                        label = scales::comma) +
  theme(legend.position = "right")

# text(sort(top.ten2), bp, labels =  as.character(round(sort(top.ten2), 2)), cex = .75, pos = 3)

## Race rate by state
######################
native <- tapply(census.subset$Native, list(census.subset$state), mean)
top.five <- head(sort(native, decreasing = T), 5)

# plotting top 5 states with the highest rate of Natives
bp1 <- barplot(top.five, beside = T, col = "orange", 
        main = "Top 5 states with highest percentage of Native Americans", 
        border = F, family = "mono", cex.main = .90)
text(bp1, 0, as.character(round(sort(top.five, decreasing = T), 2)), cex = .75, pos = 3)

## Diversity vs Income
######################
census.subset %>%
  group_by(State, state.abb) %>%
  dplyr::summarise(Income = mean(Income),
                   TotalPop = sum(TotalPop),
                   Hispanic = mean(Hispanic),
                   White = mean(White),
                   Black = mean(Black), 
                   Native = mean(Native),
                   Asian = mean(Asian), 
                   Pacific = mean(Pacific)) %>%
  {. ->> z} # saving the result of piping

# adding diversity score to the dataframe with ethnicity and income
income.diversity <- as.data.frame(z) # first convert table to dataframe
income.diversity <- cbind(income.diversity, D.Score = diversify(income.diversity))

# sorting from the most diverse state to the least
sorted.income.diversity <- income.diversity[order(income.diversity$D.Score, decreasing = T), ]
plot(sorted.income.diversity$Income, sorted.income.diversity$D.Score, col = "orange", pch = 20,
     main = "Income vs Diversity",
     xlab = "Income",
     ylab = "Diversity Score",
     family = "mono")
# add a trend line
abline(lm(sorted.income.diversity$D.Score ~ sorted.income.diversity$Income))

# Plotting most ethnically diverse states: top 5
diversity.by.state <- tapply(sorted.income.diversity$D.Score, list(sorted.income.diversity$state.abb), mean)
top.five2 <- head(sort(diversity.by.state, decreasing = T), 5)
bp3 <- barplot(top.five2, beside = T, col = "orange", 
        main = "Top 5 ethnically diverse states", 
        border = F, family = "mono", cex.main = .90, ylim = c(0, 0.8))
text(bp3, 0, as.character(round(top.five2, 2)), cex = .75, pos = 3)

# US map with Poverty Rate
##########################
boxplot(census.subset$Income, col = "#2B2BC9", ylim = c(0, 250000)) # color used for all plots in AI

