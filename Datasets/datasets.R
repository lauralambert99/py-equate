# Gather Datasetss
setwd("/Users/bao2yx/Library/CloudStorage/OneDrive-JamesMadisonUniversity/Research/Py-equate/Datasets")

library(equate)
library()

# ACT math
data("ACTmath")
write.csv(ACTmath, "ACTmath.csv")

# KBneat
data("KBneat")
write.csv(KBneat$x, "KBneatx.csv")
write.csv(KBneat$y, "KBneaty.csv")

# PISA 2009 USA
data("PISA")
write.csv(PISA$students, "PISAstudents.csv")
write.csv(PISA$booklets, "PISAbooklets.csv")
write.csv(PISA$items, "PISAitems.csv")
# write.csv(PISA$totals, "PISAtotals.csv") contains 13 forms and cannot be written in csv file
save(PISA, file = "PISA.RData")

# From SNSequate
library(SNSequate)

# CB data
data("CBdata")
write.csv(CBdata$datx1y2, "CBdatax1y2.csv")
write.csv(CBdata$datx2y1, "CBdatax2y1.csv")

# Math20EG
data("Math20EG")
write.csv(Math20EG, "Math20EG.csv")
data("Math20SG")
write.csv(Math20SG, "Math20SG.csv")

# IRT from Equating in R textbook

load(url("http://www.mat.uc.cl/~jorge.gonzalez/EquatingRbook/ADMneatX.Rda"))
load(url("http://www.mat.uc.cl/~jorge.gonzalez/EquatingRbook/ADMneatY.Rda"))
write.csv(ADMneatX,"ADMneatX.csv")
write.csv(ADMneatY, "ADMneatY.csv")
