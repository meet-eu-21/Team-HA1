#!/bin/bash

while getopts ":j:" OPTION; do
    case $OPTION in
    j)
        JSON=$OPTARG
        echo "$JSON"
        if [[ ! -f "$JSON" ]]; then
            echo "The provided JSON file does not exist."
            exit 1
        fi
        ;;
    *)
        echo "Incorrect options provided"
        exit 1
        ;;
    esac
done

source /home/arnoldtl/.bashrc

conda activate meeteu_arnoldt_kittner2

bash ../../preprocessing/arrowhead_solution.py -d "$JSON"

exit 0;