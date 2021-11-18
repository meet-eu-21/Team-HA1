def extract_bins(graph_matrices, chromosome_list_graph_matrices):
    '''

    :param graph_matrices: adjacency matrices representing Hi-C graphs for a specific chromosome and cell line
    :param chromosome_list_graph_matrices: list of chromosome names for each graph in graph_matrices
    :return bins_chromosomes: bins for each chromosome
    '''

    bins_chromosomes = []

    for chromsome, graph in zip(chromosome_list_graph_matrices, graph_matrices):
        graph.rows

