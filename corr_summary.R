library(corrplot)

train <- read.csv('input/train.csv')
X <- train
X$Id <- NULL
X$SalePrice <- NULL

X.numeric <- X[, sapply(X, is.numeric)]
X.scale <- scale(X.numeric, center=TRUE, scale=TRUE)

X.scale[is.na(X.scale)] <- 0

X.cor <- cor(X.scale)
corrplot(X.cor, order = "hclust")

library(mice)
md.pattern(train)

library(VIM)
aggr_plot <- aggr(train, col=c('navyblue','red'), 
                  numbers=TRUE, sortVars=TRUE, 
                  labels=names(train), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
