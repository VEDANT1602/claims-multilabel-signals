import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,f1_score
from core import FEATURES,LABELS,probabilities,choose_thresholds,decisions,binary_f1
from generate import generate,arrays
from support import ROOT,dump,manifest

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=generate()
    xtrain,ytrain,train=arrays(rows,'train');xval,yval,val=arrays(rows,'validation');xtest,ytest,test=arrays(rows,'test')
    groups={s:{r['family'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    scaler=StandardScaler().fit(xtrain);models=[]
    for j in range(len(LABELS)):models.append(LogisticRegression(C=1,max_iter=300,random_state=1602).fit(scaler.transform(xtrain),ytrain[:,j]))
    artifact={'features':FEATURES,'labels':LABELS,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
              'coef':[m.coef_[0].tolist() for m in models],'intercept':[float(m.intercept_[0]) for m in models]}
    pval=probabilities(xval,artifact);thresholds,audit=choose_thresholds(yval,pval)
    artifact['thresholds']=thresholds;dump(ROOT/'models/linear-heads.json',artifact)
    ptest=probabilities(xtest,artifact);pred=decisions(ptest,thresholds);supported=[j for j,t in enumerate(thresholds) if t is not None]
    fixed=(ptest>=.5).astype(int);ys=ytest[:,supported];ps=pred[:,supported];fs=fixed[:,supported]
    bylabel=[]
    for j,label in enumerate(LABELS):
        bylabel.append({'label':label,'test_positive':int(ytest[:,j].sum()),'test_ap':float(average_precision_score(ytest[:,j],ptest[:,j])),
                        'threshold':thresholds[j],'decision_status':'supported' if j in supported else 'withheld',
                        'fixed_f1':float(f1_score(ytest[:,j],fixed[:,j],zero_division=0)) if j in supported else None,
                        'selected_f1':float(f1_score(ytest[:,j],pred[:,j],zero_division=0)) if j in supported else None})
    slices={}
    for name in ['reference','shifted']:
        mask=np.array([r['slice']==name for r in test]);slices[name]={'events':int(mask.sum()),
          'macro_ap_supported':float(average_precision_score(ytest[mask][:,supported],ptest[mask][:,supported],average='macro')),
          'selected_macro_f1':float(f1_score(ytest[mask][:,supported],pred[mask][:,supported],average='macro',zero_division=0))}
    # Family bootstrap holds the fitted models and validation thresholds fixed.
    rng=np.random.default_rng(42);differences=[]
    for _ in range(500):
        picks=rng.integers(0,200,size=200);indices=(picks[:,None]*4+np.arange(4)).ravel()
        differences.append(float(np.mean(binary_f1(ys[indices],ps[indices]))-np.mean(binary_f1(ys[indices],fs[indices]))))
    metrics={'records':len(rows),'split_records':{'train':len(train),'validation':len(val),'test':len(test)},
             'split_families':{s:len(g) for s,g in groups.items()},'supported_decision_labels':[LABELS[j] for j in supported],
             'fixed_macro_f1':float(f1_score(ys,fs,average='macro',zero_division=0)),
             'selected_macro_f1':float(f1_score(ys,ps,average='macro',zero_division=0)),
             'selected_micro_f1':float(f1_score(ys,ps,average='micro',zero_division=0)),
             'supported_macro_ap':float(average_precision_score(ys,ptest[:,supported],average='macro')),
             'macro_f1_difference_family_bootstrap_95':[float(v) for v in np.quantile(differences,[.025,.975])],
             'per_label':bylabel,'slices':slices,
             'scope':'Original synthetic operational review signals; no fraud capture or insurance decision claim.'}
    dump(output/'threshold-selection.json',{'validation_only':True,'minimum_positive_and_negative_support':25,'labels':audit})
    dump(output/'metrics.json',metrics)
    raw=[]
    for i,row in enumerate(test):
        result={'id':row['id'],'family':row['family'],'slice':row['slice']}
        for j,label in enumerate(LABELS):result.update({label+'_truth':int(ytest[i,j]),label+'_probability':float(ptest[i,j]),label+'_selected':int(pred[i,j]),label+'_fixed':int(fixed[i,j])})
        raw.append(result)
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]),lineterminator='\n');w.writeheader();w.writerows(raw)
    report=f"# Review reasons are not one binary score\n\nThe actual CPU run fits four independent logistic heads on 2,000 records from 500 families, selects thresholds on 600 records from 150 separate families, then evaluates 800 records from 200 unseen families. Decisions are supported for {len(supported)} of 4 labels; the rare label is withheld when validation support is below 25 positives or 25 negatives. It remains a separately labeled probability diagnostic, not an admitted decision.\n\nOn the same supported-label population, a fixed 0.5 threshold has macro F1 {metrics['fixed_macro_f1']:.4f}; validation-selected thresholds have macro F1 {metrics['selected_macro_f1']:.4f}, micro F1 {metrics['selected_micro_f1']:.4f}, and macro AP {metrics['supported_macro_ap']:.4f}. The paired family-bootstrap interval for macro F1 difference is {metrics['macro_f1_difference_family_bootstrap_95']}; fitted models and thresholds are held fixed.\n\nMy implementation separates each label's evidence and admission rule, commits portable learned coefficients and per-label raw predictions, and reports the generator's changed-relation slice. Reference/shifted supported macro AP is {slices['reference']['macro_ap_supported']:.4f}/{slices['shifted']['macro_ap_supported']:.4f}; the shift is deliberately synthetic and not a real drift estimate. A label with weak support should enter review rather than disappear inside a favorable micro-average.\n\nThe aggregate improvement is not universal: document_gap F1 falls from {bylabel[0]['fixed_f1']:.4f} to {bylabel[0]['selected_f1']:.4f}. That tradeoff remains in the per-label table.\n\nThe source topic was multi-label learning; this project contains no medical/drug data or real claims. The equation is simple, reasons are synthetic, and random-family separation does not establish deployment generalization. The next step is an independently curated and authorized label set with costs and review capacity, not a production eligibility rule.\n"
    (output/'report.md').write_bytes(report.encode());manifest(output);return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');a=p.parse_args();print(run(ROOT/a.output))
