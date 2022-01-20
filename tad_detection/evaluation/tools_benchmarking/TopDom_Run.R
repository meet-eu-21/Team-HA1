#Download the TopDom package

# getwd()
# setwd("/home/charlotte/TopDom/R")   #set working directory to TopDom Package
# source("TopDom_0.0.2.R")        # call on the script TopDom_0.0.2.R
################## DASMUSST DU LOESEN
setwd("/gpfs/bwfor/work/ws/hd_vu199-meeteu_files/TopDom_Results/")

#INPUT EXAMPLE: "home/charlotte/adj_matrix_cell_Rline/resolution/resolution_chrnum_adj.csv"
#OUTPUT EXAMPLE: "home/charlotte/TopDomResults/cell_line/resolution/resolution_chrnum"
library("TopDom")

resolutions = c("25kb", "100kb")
cell_lines = c("HMEC", "HUVEC", "IMR90", "NHEK", "GM12878")
ws = 5

base_input_folder = "/gpfs/bwfor/work/ws/hd_vu199-meeteu_files/TopDom_Results/"
base_output_folder = "/gpfs/bwfor/work/ws/hd_vu199-meeteu_files/TopDom_Results/"
for (cell_line in cell_lines){
    sub_folder = paste("adj_matrix_", cell_line, sep="")
    cell_line_folder = paste(base_input_folder, sub_folder, sep="")

    cell_line_output_folder = paste(base_output_folder, cell_line, sep="")
    dir.create(cell_line_output_folder)

    for (resolution in resolutions){
        resolution_folder = paste(cell_line_folder, resolution, sep="/")

        resolution_output_folder = paste(cell_line_output_folder, resolution, sep="/")
        dir.create(resolution_output_folder)

        for (chrnum in 1:23){
            name = paste(resolution, "_chr", sep="")
            output_filename = paste(name, chrnum, sep="")

            if (chrnum == 23){ 
                output_filename = paste(name, "X", sep="")
            } 
            input_filename = paste(output_filename, "_adj.csv", sep="")

            output_binadress = paste(resolution_output_folder, output_filename, sep="/") # output file name
            print(output_binadress)
            input_matrixfile = paste(resolution_folder, input_filename, sep="/") # input file name
            print(input_matrixfile)

            # TopDom_0.0.2(matrix.file = input_matrixfile, window.size = ws, outFile = output_binadress, statFilter=T)
            data = readHiC(file=input_matrixfile)
            print('data loaded')
            dir.exists(resolution_output_folder)
            fit = TopDom(data = data, window.size = ws, outFile = output_binadress, statFilter=T)
            print('TopDom result saved')
            # data needs to come from readHiC(jfdlkfskl)
        }
    }
}
