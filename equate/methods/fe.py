# -*- coding: utf-8 -*-
"""
Created on Wed May 14 07:06:55 2025

@author: laycocla
"""
import pandas as pd
import numpy as np

def fe(x, y, common_x, common_y, scores, w1):
    """
  Perform Frequency Estimation equating.

  Parameters:
  x, y: Array of raw scores for Form X and Form Y
  common_x, common_y: Arrays of anchor scores for each form
  scores: Array of score range to equate
  w1: Weight for group 1

  Returns:
  DataFrame of equated scores
  """


#Reproduce data from book
#NEED: freqtab table.  That's what this is.
x = list(range(6))
v0 = [0.04, 0.04, 0.06, 0.03, 0.02, 0.01]
v1 = [0.04, 0.08, 0.12, 0.12, 0.03, 0.01]
v2 = [0.02, 0.02, 0.05, 0.05, 0.04, 0.02]
v3 = [0.00, 0.01, 0.02, 0.05, 0.06, 0.06]

# Create the DataFrame
kb5_1 = pd.DataFrame({
    'x': x,
    'v0': v0,
    'v1': v1,
    'v2': v2,
    'v3': v3
})

#Calculate f1_x - marginal distribution of x
f1_x = kb5_1.iloc[:, 1:5].sum(axis=1)

#Calculate h1_v - another marginal distribution, v this time
h1v <- colSums(kb5_1)

#And cumulative distribution
F1x <- cumsum(f1_x)

#Calculate conditional distributions
#Don't want to sweep first column - those are the x values
cond_dist_x <- kb5_1
cond_dist_x[, -1] <- sweep(kb5_1[, -1], 2, h1v[-1], "/")

#Add in h1v row
cond_dist_x <- rbind(cond_dist_x, h1v)

#And fix the sum of x values - replace with "h1(v)
cond_dist_x[7, 1] <- "h1(v)"


KB 5.2
To calculate the joint distribution of Form X in Population 2, take the conditional distribution of Form X in Population 1 and multiply it by $h_{2}(v)$.  These values were obtained from table 5.2.
```{r}
#| echo: FALSE
#Pull these values from table 5.2
h2v <- c(0.20, 0.20, 0.40, 0.20)

#Need to multiply conditional distribution table from 5.1 by h2v
cond_dist_x2 <- cond_dist_x
cond_dist_x2[, -1] <- sweep(cond_dist_x[, -1], 2, h2v, "*")

#Remove h1v row, replace with h2v row
cond_dist_x2 <- cond_dist_x2[-7, ]

h2v_a <- c(NA, 0.20, 0.20, 0.40, 0.20)
cond_dist_x2 <- rbind(cond_dist_x2, h2v_a)

#Rename empty cell
cond_dist_x2[7,1] <- "h2(v)"

#Add marginal distribution of x
f2_x <- rowSums(cond_dist_x2[c(1:6), c(2:5)])

#Add a dummy space
f2_x <- c(f2_x, " ")

#Add f2x to dataframe
cond_dist_x2 <- cbind(cond_dist_x2, f2_x)

#Also adding cumulative distribution
#It's in table 5.4
F2x <- cumsum(f2_x)

#Add a dummy space to be able to add it to the dataframe, then add
F2x[7] <- " " #NOTE: doing this changes the entire vector to a character vector!  Don't think I care for now.

cond_dist_x2 <- cbind(cond_dist_x2, F2x)

#Now, make the pretty table
ft2 <- flextable(cond_dist_x2)

ft2 <- hline(ft2, i = 6, part = "body") |>
  add_header_row(values = c("", "v", ""), colwidths = c(1, 4, 2)) |>
  align(align = "center", part = "header")
ft2
```

KB 5.3

To find the form Y equipercentile equivalents of Form X scores, we will first need to calculate the synthetic population.

```{r}
#| echo: FALSE
f1x <- f1_x
F1x <- cumsum(f1x)
f2x <- as.numeric(f2_x[1:6])
F2x <- cumsum(f2x)

g1y <- c(0.105, 0.215, 0.210, 0.225, 0.145, 0.10)
G1y <- cumsum(g1y)
g2y <- c(0.08, 0.20, 0.22, 0.25, 0.15, 0.10)
G2Y <- cumsum(g2y)

w1 <- 0.5
w2 <- 0.5

#Marginal distributions of synthetic population
fsx <- w1*f1x + w2*f2x
gsy <- w1*g1y + w2*g2y

#Cumulative distributions of synthetic population
Fsx <- cumsum(fsx)
Gsy <- cumsum(gsy)

#Percentile rank functions
Psx <- 100*(dplyr::lag(Fsx, default = 0) + (fsx/2))
Qsy <- 100*(dplyr::lag(Gsy, default = 0) + (gsy/2))

#Put all calculations together in a dataframe
equi <- data.frame(x = 0:5,
                   fsx = fsx,
                   Fsx = Fsx,
                   Psx = Psx,
                   gsy = gsy,
                   Gsy = Gsy,
                   Qsy = Qsy)

#Do the equating (thanks HW1!!)
#First, make G(y) * 100 value column - easier to reference this
equi$Gsy_100 <- equi$Gsy*100

#Make a function to take each P(x) value and find the smallest Gy_100 value that is => to it
#Use one of the apply commands?

#Started by finding  smallest GY_100 >= Px
find_gte_Gsy <- function(Psx, Gsy_100) {
  # Filter the values in Gy_100 that are greater than or equal to the current P value
  Gsy_gte <- Gsy_100[Gsy_100 >= Psx]
  
  # Return the smallest value, or NA if none are valid
  if(length(Gsy_gte) > 0) {
    return(min(Gsy_gte))
  } else {
    return(NA)  # In case no value in G is greater than or equal to the current P value
  }
}

equi$min_Gsy <- sapply(equi$Psx, find_gte_Gsy, Gsy_100 = equi$Gsy_100)

#But what we really want is the corresponding Y value
#Edit above to give us that
find_Y_star <- function(Psx, Gsy, Y) {
  gte <- which(Gsy >= Psx) #This is pulling the indices of Gy_100 values >= the P(x)
  
  if(length(gte) > 0) {
    Gsy_index <- gte[which.min(Gsy[gte])] #Then only pull the smallest of the values
    return(Y[Gsy_index]) #But we want the corresponding Y value, so use that index to pull the Y
  } else {
    return(NA) #Should this be NA?  Or 100?  Check on that.
  }
}

#Use the function to make a new column 
equi$Y_star_u <- sapply(equi$Psx, find_Y_star, Gsy = equi$Gsy_100, Y = equi$x)

#Make a G(Y*u) column? Otherwise, how to use Y*u as a reference for Gy?
#Maybe another function using indexing like above?
find_GsY_star <- function(Y_star, Y, Gsy) {
  get_index <- match(Y_star, Y) #First, match the Y* value with Y, get that index
  
  if(!is.na(get_index)) {
    return(Gsy[get_index])  #Then, using that index of Y, get the corresponding Gy
  }
  else{
    return(NA)
  }
}

#Make the GY* column
equi$Gsy_star <- sapply(equi$Y_star_u, find_GsY_star, Y = equi$x, Gsy = equi$Gsy)

#Can't just lag it in the equation, because some are repeats - a simple one step back won't work.
#Will also need a lag Y*u for equation
#Use above, just lag it?
find_GsY_star_lag <- function(Y_star, Y, Gsy) {
  get_index <- match((Y_star-1), Y) #First, match the lag Y* value with Y, get that index
  
  if(!is.na(get_index)) {
    return(Gsy[get_index])  #Then, using that index of Y, get the corresponding Gy
  }
  else{
    return(0) #The early ones were throwing NA - switched this to return zero
  }
}

equi$Gsy_star_lag <- sapply(equi$Y_star_u, find_GsY_star_lag, Y = equi$x, Gsy = equi$Gsy)

#Now, calculate e(y)x
equi$eys_x <- ((((equi$Psx/100) - equi$Gsy_star_lag)/(equi$Gsy_star - equi$Gsy_star_lag)) + (equi$Y_star_u - 0.5))

#Display results in a table
ft3 <- flextable(equi[,c(1:7, 13)])
ft3
