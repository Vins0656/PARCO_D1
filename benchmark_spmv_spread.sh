#!/bin/bash
set -euo pipefail

# File di output
OUTPUT_1="risultati_1_spread.txt"
OUTPUT_2="risultati_2_spread.txt"
: > "$OUTPUT_1"
: > "$OUTPUT_2"

# Input
FILES_1=("wiki-talk-temporal.mtx" "kron_g500-logn16.mtx" "t2em.mtx"  "engine.mtx")
FILES_2=( "rail2586.mtx" "GL7d17.mtx" "GL7d21.mtx" "rel9.mtx")
THREADS=(1 2 4 8 16 32 64)

# Sorgenti/eseguibili
SOURCE_1="parcoD1.c++"
SOURCE_2="parcoD1Lib.c++"
EXEC_NO_OPT="a_no_opt.exe"
EXEC_OPT="a_opt.exe"

# Compilazioni silenziose
g++ -std=c++11 -fopenmp "$SOURCE_1" "$SOURCE_2" -o "$EXEC_NO_OPT"
g++ -std=c++11 -O3 -march=native  -fopenmp "$SOURCE_1" "$SOURCE_2" -o "$EXEC_OPT"

# Funzione: 1 run per configurazione
run_tests() {
  local exec_name=$1
  local output_file=$2
  shift 2
  local files_to_run=("$@")

  for file in "${files_to_run[@]}"; do
    for t in "${THREADS[@]}"; do
      export OMP_NUM_THREADS=$t
      export OMP_PLACES=cores
      export OMP_PROC_BIND=spread
      export OMP_DISPLAY_ENV=FALSE
      export OMP_DISPLAY_AFFINITY=TRUE
      export OMP_AFFINITY_FORMAT="thread %i -> %A"

      tmp_all="$(mktemp)"
      "./$exec_name" "$file" > "$tmp_all" 2>&1 || true

      {
        echo "Matrix: $(basename "$file")"
        echo "Threads: $t | Iterations: ${NUM_IT:-10} | Chunk: ${CHUNK:-1}"
        # 2) Risultati stampati dal programma: numeri e parole chiave comuni
        awk '
          /^[[:space:]]*[0-9]+([[:space:]]+[0-9]+)*[[:space:]]*$/ { print; next }
          /Average|mean|tempo|time|ms|GFLOP|GB\/s|MB\/s/ { print }
        ' "$tmp_all" || true
        echo
      } >> "$output_file"

      rm -f "$tmp_all"
    done
  done
}

# Esecuzioni
# run_tests "$EXEC_NO_OPT" "$OUTPUT_1" "${FILES_1[@]}"
# run_tests "$EXEC_NO_OPT" "$OUTPUT_2" "${FILES_2[@]}"
run_tests "$EXEC_OPT"    "$OUTPUT_1" "${FILES_1[@]}"
run_tests "$EXEC_OPT"    "$OUTPUT_2" "${FILES_2[@]}"
