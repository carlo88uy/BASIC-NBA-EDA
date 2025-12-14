library(car)
library(MASS)
# Reading data into 
df = read.csv("NBA 2023-2024 Dataset (Combined) V2 - Sheet1.csv")
attach(df)
df$Attend..G <- as.numeric(gsub(",", "", df$`Attend..G`))
cols <- c("X3PA", "BLK", "PTS", "W", "Attend..G", "All.NBA.Player")
data <- df[, cols]

# correlation matrix
cor(data)

# Graphics
# histogram for wins
hist(W)
# histogram for 3 point attempts
hist(X3PA)
# Histogram for blocks
hist(BLK)

# Histogram for points
hist(PTS)
hist(Attend..G)

# Bar chart for All-NBA Player counts
barplot(table(All.NBA.Player),
        xlab = "All NBA Player",
        ylab = "Frequency")
par(mfrow = c(2, 3))
# Boxplot for 3 point attempts
boxplot(data$X3PA, xlab = "3-Point Attempts")

# Box plot for blocks
boxplot(data$BLK, xlab = "Blocks")

# Box plot for points
boxplot(data$PTS, xlab = "Points")

# Box plot for wins
boxplot(data$W, xlab = "Wins")

# Box plot for attendace per game
boxplot(data$Attend..G, xlab = "Attendance")

par(mfrow = c(1, 1))


# Finding typical values

options(scipen = 20)
# mean, median, standard deviation
mean_values <- sapply(data, mean, na.rm = TRUE)
median_values <- sapply(data, median, na.rm = TRUE) 
sd_values <- sapply(data, sd, na.rm = TRUE)       

# makes a table of typical values for each variable
typical_values <- data.frame(
  Mean = mean_values,
  Median = median_values,
  Standard_Deviation = sd_values
)
print(typical_values)

full_model <- lm(Attend..G ~., data = data)
summary(full_model)

# Partial regression plots
crPlots(full_model)


#confounding test

model <- lm(Attend..G ~ PTS + W + I(W^2) + PTS*W + PTS*I(W^2), data = data)
summary(model)
model1 <- lm(Attend..G ~ W, data = data)
summary(model1)
model2 <- lm(Attend..G ~ W + PTS, data = data)
summary(model2)
model_a <- lm(Attend..G ~ I(W^2), data = data)
model_b <- lm(Attend..G ~ I(W^2) + PTS, data = data)

summary(model_a)
summary(model_b)

#shapiro wilk test
shapiro.test(full_model$residuals)


# fitted values vs. jackknife residuals
plot(full_model$fitted.values, studres(full_model))
abline(h = 0)



final_model <- lm(Attend..G ~ W + I(W^2),data = data)

# collinearity check
library(olsrr)
options(scipen = 20)
ols_coll_diag(full_model)

# standardized variables
data$X3PA <- scale(data$X3PA)
data$PTS <- scale(data$PTS)

full_model <- lm(Attend..G~., data = data)
ols_coll_diag(full_model)


