"""Apply selected checkpoints to the frozen later cohort; never fit here."""
import argparse
import json
from pathlib import Path
from data_protocol import read_json,write_json
from release_common import sha
from later_scope import select_runs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--later',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=['paper','all'],default='paper',
                   help='paper requires exactly the 84 paper combinations; all evaluates all supplied completed runs')
    p.add_argument('--dry-run',action='store_true',help='validate selection/checkpoint presence without inference or output writes')
    args=p.parse_args()
    selected,audit=select_runs(args.runs,args.scope)
    if args.output.exists():raise ValueError('Output directory must be new: '+str(args.output))
    if args.dry_run:
        print(json.dumps(audit,indent=2));return
    import torch,joblib
    from experiment import PrefixDataset,baseline_features
    from train_neural import evaluate,on_device,save_predictions
    from models import DotaMultiModalPredictor
    from evaluation import probability_metrics
    meta=read_json(args.later/'metadata.json')
    for name,expected in meta['feature_files'].items():
        if sha(args.later/Path(name).name)!=expected:raise ValueError('Later feature hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'evaluation_scope.json',audit)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count=0
    for path,result,weight in selected:
        minute=result['minute'];name=result['model']
        dataset=PrefixDataset(args.later/f'{minute}m.npz')
        if weight.suffix=='.pt':
            saved=torch.load(weight,map_location='cpu',weights_only=True)
            model=DotaMultiModalPredictor(**saved['model_config']).to(device)
            model.load_state_dict(saved['state_dict'])
            probs=evaluate(model,on_device(dataset,device),device,minute+1)
        else:
            model=joblib.load(weight)
            probs=model.predict_proba(baseline_features(dataset,simple=name=='gold_logistic'))[:,1]
        folder=args.output/path.parent.name;folder.mkdir()
        save_predictions(folder/'predictions.npz',dataset,probs)
        write_json(folder/'result.json',{'model':name,'minute':minute,'seed':result['seed'],'evaluation_scope':args.scope,
              'checkpoint_sha256':sha(weight),'metrics':probability_metrics(dataset.arrays['outcome'],probs),
              'interpretation':'retrospective previously inspected cohort; no tuning'})
        count+=1
    if not count:raise ValueError('No completed runs found')
    print('Evaluated',count,'selected runs')

if __name__=='__main__':main()
