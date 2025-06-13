#!/bin/bash

INPUT="public_data.csv"
INPUT_pr="private_data.csv"

EVAL="eval.py"
RESULT="public_result.csv"
RESULT_pr="private_result.csv"
METHODS=("kmeans" "kmeans++" "minibatch" "kmedoids" "gmm" "spectral" "dbscan" "agglo" "birch" "optics")
# METHODS=("dbscan" "agglo" "birch" "optics")
# METHODS=("kmeans" )

for method in "${METHODS[@]}"
do
  echo "Running $method..."

  # 刪除舊檔，避免衝突
  if [ -f "$RESULT" ]; then
    rm "$RESULT"
  fi

  # 執行分群，結果都輸出為 result.csv
  python main.py --public_input "$INPUT"   --private_input "$INPUT_pr" --method "$method" 

  # 執行評分
  echo "Evaluating $method result:"
  python $EVAL
  echo ""

  # 刪除結果，保持環境乾淨
  rm "$RESULT"
  rm "$RESULT_pr"
done
