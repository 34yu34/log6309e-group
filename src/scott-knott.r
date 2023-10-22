library(ScottKnott) 
data(CRD1) 

#From:formula 

#Simple! 

sk1<-SK(y~x, data=CRD1$dfm, which='x') boxplot(sk1)