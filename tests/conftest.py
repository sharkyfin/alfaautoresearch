import os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
