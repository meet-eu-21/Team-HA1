# Evaluation

An exact description of the scripts to run the tools for benchmarking can be found in the folder ```./tools_benchmarking``` and the associated [README](tools_benchmarking/README.md).

# Evaluation pipeline

To use the evaluation script please run ```evaluate.py``` with the argument ```--path_parameters_json```. We do provide a sample ```parameters.json``` file in the folder.

## What the pipeline does:

Generally, the evaluation script fulfills the following tasks:

1. Loading of TAD results: Loading of results from TAD calling tools like the ones we develop or Arrowhead or comparable tools. Here, the folder `./MeetEU/tad_detection/evaluation/results/` is important. It contains the TAD prediction results of different tools can be found. Currently, it contains the TADs called by Arrowhead (`Arrowhead_GM12878_100kb_dict.p`) and TopDom (`TopDom_GM12878_100kb_dict.p`). As soon as it is available we will add the corresponding .p file for our own tool. Also, we will add the results of another benchmarking tool, which should not be of interest for you, as this is a TAD calling tool using also a graph-based structure. The structure of the dict.p files is described in the appendix below. This is of interest of you, because you need to format your results in this structure, so you can run them through the pipeline.
1. Venn-diagram visualization: For all inputted .p files the Venn-Diagramm of all methods together is created and saved.
1. Jaccard index calculation: For all inputted .p files the Jaccard index between each of the .p files is calculated separately and saved.
1. TAD region size calculation: Statistics on the size of TADs (Mean, Max, Min, quantiles) are calculated per chromosome and for all chromosomes together. The results are saved.

## Setting the parameters

The `parameters.json` file is your control center. To run the evaluation pipeline you need to adjust several parameters.
We provide an overview about the parameters:

1. "cell_line": The cell line for which you have run the TAD calling algorithm, e.g. GM12878
1. "paths_predicted_tads_per_tad_prediction_methods"</li> The paths of .p files with your TAD calling results. Please replace the paths in the file with the exact paths of the files in the folder `./MeetEU/tad_detection/evaluation/results/`.
1. "output_directory": Here outputs like the Venn-Diagrams and the statistics on the TAD region sizes are saved.

## Appendix:

1. Structure of .p files: In the meeting you told us, that you may only have the boundaries of the TADs. The pipeline is currently not adapted to TADs, where only the boundaries are known. If this issue still consists on your side, please write us, so we can adapt the pipeline accordingly. At the moment, each method needs to have its own dict for each cell line. The dict is a nested dict:
   ```python
   {'methodnamestring': {'chrnumstring': [list of tad regions], 'chrnumstring': [list of tad regions]}}
   ```
3. If there are any questions please contact Lucas or Charlotte on Discord.

