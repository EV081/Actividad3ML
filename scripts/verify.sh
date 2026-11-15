#!/usr/bin/env bash
# Verificaciones del proyecto. Uso: scripts/verify.sh [tests|notebooks|outputs|doc|all]
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
fail=0
ok(){ echo "OK   $*"; }
bad(){ echo "FAIL $*"; fail=1; }
what=${1:-all}

check_git(){
  bad_files=$(git status --porcelain | grep -E '(jc/|teoria/| M data/|D data/)' || true)
  [ -z "$bad_files" ] && ok "git: no se tocan jc/ data/ teoria/" || bad "git toca zonas prohibidas: $bad_files"
}
check_tests(){
  if ls tests/test_*.py >/dev/null 2>&1; then
    $PY -m pytest -q tests && ok "pytest" || bad "pytest"
  else bad "no hay tests"; fi
}
check_notebooks(){
  for nb in notebooks/clase/*.ipynb notebooks/experimentacion/*.ipynb; do
    [ -e "$nb" ] || continue
    $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb" >/dev/null 2>&1 \
      && ok "nb $nb" || bad "nb $nb"
  done
}
check_outputs(){
  $PY - <<'PY' || fail=1
import glob, sys, numpy as np
sys.path.insert(0, "src")
from pcseg.io import read_ply
N = 1647021; bad = False
files = glob.glob("entregable/*.ply")
if not files: print("FAIL no hay PLY en entregable/"); sys.exit(1)
for f in files:
    p = read_ply(f)
    n = len(p["xyz"]); ncol = len(np.unique(p["rgb"], axis=0))
    good = n == N and ncol > 3
    print(("OK   " if good else "FAIL ") + f"{f}: {n} puntos, {ncol} colores distintos")
    bad |= not good
sys.exit(1 if bad else 0)
PY
}
check_doc(){
  ( cd doc && tectonic informe.tex >/dev/null 2>&1 ) && ok "tectonic compila" || bad "tectonic falla"
  [ -f doc/informe.pdf ] && pdfinfo doc/informe.pdf | grep Pages || bad "sin informe.pdf"
}
check_git
case $what in
  tests) check_tests;; notebooks) check_notebooks;; outputs) check_outputs;; doc) check_doc;;
  all) check_tests; check_notebooks; check_outputs; check_doc;;
esac
exit $fail
