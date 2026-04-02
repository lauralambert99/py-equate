# pyequate

pyequate is an open-source package containing different test score equating methods in one location.  It includes methods for random groups, common-item/non-equivalent groups, and IRT equating.  The version on GitHub will be the most current, with updates pushed to pyPI.

## Areas Currently Under Development

- Standard errors of equating for common-item/non-equivalent groups methods
- pre- and post-smoothing for equipercentile equating

## Required Packages

pyequate requires `NumPy`, `Pandas`, `SciPy`, and `Statsmodels`.

## Public Datasets

A number of publicly available datasets are included within this GitHub page to allow testing of functions.  

ACTmath

  - ACT Mathematics Test Scores
  - score distributions of two form of the ACT math test, as presented in table 2.5 of **Test Equating, Scaling, and Linking** (Kolen and Brennan, 2004; p. 50)
  - These data are also provided with the equating software RAGE from the University of Iowa (<https://education.uiowa.edu/casma/computer-programs>) as well as in the `Equate` package in R
  
ADMneatX and ADMneatY



CBdatax1y2 and CBdatax2y1

  - Data from small field study from international testing program (Von Davier et al., 2004)
  - Two different tests: X (75 items) and Y (76 items)
  - independent, random samples from single population took each form
  - X1Y2: students took form X first, then form Y
  - X2Y1: students took form Y first, then form X
  - Also contained in the `SNSequate` package in R

KBneatx and KBneaty

  - Contains scores for two forms (x and y) of a 36-item test using nonequivalent groups with anchor items. The 12 anchor items are internal. Examples using this dataset can be seen in **Test Equating, Scaling, and Linking** (Kolen and Brennan, 2004)
  - These data are also provided with the CIPE software from the University of Iowa (<https://education.uiowa.edu/casma>) as well as in the `Equate` package in R
  

Math20EG 

  - Raw sample frequencies of number-correct scores for two parallel 20-item mathematics tests given to two samples from a national population (Holland and Thayer, 1989; Von Davier et al., 2004)
  - Also contained in the `SNSequate` package in R


Math20SG

  - Bivariate sample frequencies of number-correct scores for two parallel 20-item mathematics tests given to a single sample from a national population (Holland and Thayer, 1989; Von Davier et al., 2004)
  - Also contained in the `SNSequate` package in R
  
PISA

  - The following four datasets come from the 2009 PISA administration (Programme for International Student Assessment)
  - Available online at www.oecd.org
  - Also available in the `Equate` package in R


PISAbooklets

  - Description of 13 general cognitive assessment booklets
  - Variables:
    - `bookid`: unique test booklet ID
    - `clusterid`: cluster or subset ID in which item placed; items fully nested within clusters. Each item cluster is present in four different booklets.
    - `itemid`: unique item ID
    - `order`: order in which cluster appears in a booklet

PISAitems

  - Description of items
  - Variables:
    - `itemid`: unique item ID
    - `clusterid`: cluster or subset ID in which item placed
    - `max`: maximum possible score; values of 1 (dichotomous) or 2 are possible
    - `subject`: item subject; first character in `itemid` and `clusterid`
    - `format`: item format; mc = multiple choice; cmc = complex multiple choice; ocr = open constructed response; ccr = closed constructed response
    - `noptions`: number of options; 0 except for some mc items

PISAstudents

  - Contains item response data for 189 items in general cognitive assessment, across 165 schools and 13 different test booklets for 5233 students
  - Variables:
    - `stidstd`: unique student ID
    - `schoolid`: unique school ID
    - `bookid`: unique test booklet ID
    - `langn`: Spoken language at home; English = 313; Spanish = 156; Other = 859.
    - `m033q01` to `s527q04t`: scored item-response data
    - `pv1math` to `pv5read5`: scale scores
    
PISA.RData

  - 13 separate data frames (one per booklet)
  - Columns correspond to student total scores on each cluster of the booklet

*Sources*

Gonzalez, J. and Wiberg, M. (2017) *Applying Test Equating Methods Using R*. Springer Cham. https://doi.org/10.1007/978-3-319-51824-4

Holland, P. and Thayer, D. (1989). *The kernel method of equating score distributions.* (Technical Report No 89-84). Princeton, NJ: Educational Testing Service.

Kolen, M. J., and Brennan, R. L. (2004). *Test Equating, Scaling, and Linking*. (2nd ed.), New York:
Springer.

Von Davier, A., Holland, P., and Thayer, D. (2004). *The Kernel Method of Test Equating*. New York, NY: Springer-Verlag.