getwd()
setwd("/home/charlotte/TopDom/R")

########
# normalized HiC matrix
filechr7 <-  "/home/charlotte/chr7adj.csv"
chr = "chr7"
source("readHiC.R")
data <- readHiC(filechr7, chr = chr, binSize = 100000)

print(data$bins[1:10,]) ## a TopDomData object
# data made of bins and counts

binadress <- "/home/charlotte/TopDomTests/bin2/bintest"
domadress <- "/home/charlotte/TopDomTests/bin2/domaintest"
ws <- 5     # as is recommended

source("TopDom.R")
fit <- TopDom(data=data, window.size = ws, outBinSignal = binadress, outDomain = domadress)

###################################
# normalized HiC matrix + 3 columns 
filename <- "/home/charlotte/chr7adj3.csv"
ws <- 5     # as is recommended
binadress <- "/home/charlotte/TopDomTests/bin/bintest"

# file should be a n x n+3 matrix (chrnum, bin start, bin end, N numbers normalized value)

source("TopDom_0.0.2.R")
TopDom_0.0.2(matrix.file = filename, window.size = ws, outFile = binadress, statFilter=T)

# bed = "/home/charlotte/TopDomTests/bin/bintest.bed"
# binSignal = "/home/charlotte/TopDomTests/bin/bintest.binSignal"
# domain = "/home/charlotte/TopDomTests/bin/bintest.domain"
# read.csv(bed, sep="\t")

# source("countsPerRegion.R")
# countsPerRegion(data= read.csv(bed), regions= read.csv(bed))

