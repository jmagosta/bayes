# R goodies

The simplest ggplot example

df = data.frame(x= 1:100, y=1:100)
ggplot(df, aes(x=x,y=y)) + geom_point()