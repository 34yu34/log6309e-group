library(ScottKnottESD)


data1 <- read.csv("../data/extension2.csv")

# Using Non-Parametric ScottKnott ESD test
sk <- sk_esd(data1, version="np")
plot(sk)
