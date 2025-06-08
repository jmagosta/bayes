 # Run cart on the two variable x_or data
library("dplyr")
library("readr")
library("ggplot2")
library("rpart")
library("rpart.plot")

options(width = 250)

# Look at the data
xor_df <- readr::read_csv(file = "simulated_states.csv") %>% as.data.frame()

# plot(y ~ x1, data = xor_df)
ggplot(xor_df, aes(x = x1, y = x2, color = y)) + geom_point()

 # Baseline is a conventional OLS model
ols_xor <- lm(y ~ ., data = xor_df)
print(summary(ols_xor))

# Run a classification tree
classification_tree <- rpart(y ~ x1 + x2 + x3, data = xor_df, method="class")
# plot.new()
#rpart.plot(classification_tree)



# plot.new()
rpart.plot(classification_tree)
print(summary(classification_tree))
print(rpart.rules(classification_tree))

leaf_nodes <- classification_tree$frame %>% filter( var == '<leaf>')
print(leaf_nodes)
