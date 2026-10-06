"""One entry point for all 168 selected fits (including candidate selection)."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--minutes',nargs='+',type=int,choices=[10,20,30,40],default=[10,20,30,40])
p.add_argument('--seeds',nargs='+',type=int,choices=[17,29,43],default=[17,29,43])
args=p.parse_args()
for script in ('train_neural.py','train_trees.py','train_linear_boosting.py'):
    subprocess.run([sys.executable,str(ROOT/'code'/script),'--cache',str(ROOT/'inputs/historical'),
                    '--output',str(args.output.resolve()),'--minutes',*map(str,args.minutes),'--seeds',*map(str,args.seeds)],check=True)
