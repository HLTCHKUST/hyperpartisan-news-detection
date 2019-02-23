#!/usr/bin/env bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--input)
    inputDataset="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--output)
    outputDir="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# echo input  = "${inputDataset}"
# echo output     = "${outputDir}"

# inputDataset=$1
# outputDir=$2

for file in $inputDataset/*; do
    filename=${file##*/}
done
fullpath=$inputDataset/$filename

echo $fullpath
echo $outputDir

# python3 /home/yeon-zi/fakenews_submission/final_predict_tira.py --manual_name LstmAr_LstmTit_att_2 --save_path /home/yeon-zi/fakenews_submission/pub_model/ --input_path $fullpath --output_path $outputDir --hidden_dim=100 --use_attn
# python3 /home/yeon-zi/fakenews_submission/final_predict_tira.py --use_bert --input_path $fullpath --output_path $outputDir --batch_size 16 --hidden_dim 400 --hidden_dim_tit 100

# python3 /home/yeon-zi/fakenews_submission/final_predict_tira.py --use_bert --bert_from_scratch --input_path $fullpath --output_path $outputDir --batch_size 8 --hidden_dim 300 --hidden_dim_tit 100
# python3 /home/yeon-zi/fakenews_submission/final_predict_tira.py --input_path $fullpath --output_path $outputDir --hidden_dim=100 --hidden_dim_tit=50 --batch_size 8
python3 /home/yeon-zi/fakenews_submission/final_predict_tira.py --use_bert --bert_from_scratch --input_path $fullpath --output_path $outputDir --batch_size 8 --hidden_dim 300 --hidden_dim_tit 100

# CUDA_VISIBLE_DEVICES=6 python semeval-pan-2019-evaluator.py --inputDataset=test/inputDataset --inputRun=test/inputRun --outputDir=test/outputDir
