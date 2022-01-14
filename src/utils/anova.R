library(car)
library(dplyr)
library(ggplot2)
library(reshape2)
library(stargazer)
library(xtable)

# Computing Table 2 in Appendix C.

data = data.frame( 
    ZGO = c(0.42160729, 0.42024386, 0.42175281, 0.42039060, 0.42109039), 
    FGO_05 = c(0.42502114, 0.42510780, 0.42351663, 0.42407501, 0.42295932), 
    FGO_20 = c(0.44186696, 0.44080600, 0.44552431, 0.43781984, 0.43481400)
)

stargazer(data)

data = melt(data)
colnames(data) = c("Dataset", "Accuracy") 

p<-ggplot(data, aes(x=Dataset, y=Accuracy, fill=Dataset)) + 
    geom_boxplot() + 
    theme_minimal() + 
    scale_fill_manual(values=c("#1F78B4", "#FE7F0E", "#2BA02D")) +
    labs(title=" ",x="Generalization Level", y = "Accuracy") + 
    scale_y_continuous(labels = scales::percent) + 
    theme(legend.position="none")


ggsave("OOD_Boxplot.png", plot = p, height = 3, width = 3)

xtable(Anova(lm(data$Accuracy ~ data$Dataset)), digits=6)
