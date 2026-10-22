#!/bin/bash
# Mantiene la QOS long llena (MaxSubmitPU=4): cada vez que hay menos de 4 jobs
# long del usuario, mueve a long el ultimo job regular pendiente cuyo nombre
# empiece con $PREFIX. Termina cuando no quedan pendientes regular.
PREFIX=${1:-bc_}
while true; do
  pend=$(squeue -u $USER -h -t PD -q regular -o "%i %j" | awk -v p="$PREFIX" 'index($2,p)==1{print $1}' | sort -n)
  [ -z "$pend" ] && { echo "sin pendientes regular, fin"; exit 0; }
  nlong=$(squeue -u $USER -h -q long | wc -l)
  if [ "$nlong" -lt 4 ]; then
    j=$(echo "$pend" | tail -1)
    scontrol update JobId=$j Partition=ialab-high-unlimit,ialab-low-unlimit QOS=long \
      && echo "$(date '+%F %T') $j -> long"
  fi
  sleep 300
done
