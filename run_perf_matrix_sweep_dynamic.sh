#!/usr/bin/env bash
set -euo pipefail

# 1) Compila parcoD1_static.c++ (aggiungi/rimuovi la lib se serve)
SRC_MAIN="${SRC_MAIN:-parcoD1_dynamic.c++}"
SRC_LIB="${SRC_LIB:-parcoD1Lib.c++}"   # rimuovi se non usi la lib
EXE="${EXE:-parcoD1_dynamic.exe}"

echo "[BUILD] g++ -O3 -march=native -fopenmp ${SRC_MAIN} ${SRC_LIB} -o ${EXE}"
g++ -std=c++11 -O3 -march=native -fopenmp "${SRC_MAIN}" "${SRC_LIB}" -o "${EXE}"

# 2) Matrici e thread (come nei tuoi benchmark)
FILES_1=("wiki-talk-temporal.mtx" "kron_g500-logn16.mtx" "t2em.mtx"  "engine.mtx")
FILES_2=( "rail2586.mtx" "GL7d17.mtx" "GL7d21.mtx" "rel9.mtx")
FILES=("${FILES_1[@]}" "${FILES_2[@]}")

THREADS=(1 2 4 8 16 32 64)

# 3) Config perf e affinità
OUTDIR="${OUTDIR:-perf_out_dynamic}"
BIND="${BIND:-close}"                  # close|spread
ARGS_BASE="${ARGS_BASE:-}"             # eventuali argomenti extra all'eseguibile

export OMP_PLACES=cores
export OMP_PROC_BIND="${BIND}"
export OMP_DISPLAY_ENV=FALSE
export OMP_DISPLAY_AFFINITY=TRUE
export OMP_AFFINITY_FORMAT="thread %i -> %A"

EVENTS=(
  cycles
  instructions
  branches
  branch-misses
  cache-references
  cache-misses
  stalled-cycles-frontend
  stalled-cycles-backend
  task-clock
  context-switches
  cpu-migrations
  page-faults
)
EVSTR=$(IFS=,; echo "${EVENTS[*]}")

OUT_SUBDIR="${OUTDIR}/${EXE%.*}_${BIND}"
mkdir -p "${OUT_SUBDIR}"

SCHED_NAME="${SCHED_NAME:-dynamic}"  # Passato dal PBS o default

# Per ogni matrice: un solo file di log che contiene T=1..64 in sequenza
for MAT in "${FILES[@]}"; do
  MAT_BASENAME="$(basename "$MAT")"
  MAT_TAG="${MAT_BASENAME%.*}"
  LOG_RAW="${OUT_SUBDIR}/perf_${SCHED_NAME}_${MAT_TAG}.log"  # ← AGGIUNGI SCHED_NAME
  : > "${LOG_RAW}"

  echo "=== Matrix: ${MAT_BASENAME} ===" | tee -a "${LOG_RAW}"

  for T in "${THREADS[@]}"; do
    export OMP_NUM_THREADS="${T}"
    echo "--- EXE=${EXE} T=${T} ---" | tee -a "${LOG_RAW}"

    # 5 ripetizioni per robustezza; matrice come argomento posizionale
    perf stat -r 5 -e "${EVSTR}" -- ./"${EXE}" "${MAT}" ${ARGS_BASE} \
      >> "${LOG_RAW}" 2>&1
  done
done

echo "Done. Output in ${OUT_SUBDIR}"
