import csv
import numpy as np
from core import FEATURES,LABELS
from support import ROOT

def generate():
    rng=np.random.default_rng(1602);rows=[];group_id=0
    for split,groups in [('train',500),('validation',150),('test',200)]:
        for g in range(groups):
            latent=rng.normal();shift=split=='test' and g>=100
            for member in range(4):
                x=rng.normal(size=8)+.15*latent
                logits=np.array([1.4*x[0]-.9*x[3]+.6*x[5]-1.0,
                                 1.2*x[1]+.8*x[4]+.5*latent-.5,
                                 (-1.2 if shift else 1.2)*x[0]+.9*x[2]+.5*latent-1.3,
                                 1.1*x[6]-4.7])
                p=1/(1+np.exp(-logits));y=(rng.random(4)<p).astype(int)
                row={'id':f'family-{group_id:04d}-{member}','family':f'family-{group_id:04d}','split':split,'slice':'shifted' if shift else 'reference'}
                row.update({name:float(value) for name,value in zip(FEATURES,x)});row.update({name:int(value) for name,value in zip(LABELS,y)});rows.append(row)
            group_id+=1
    (ROOT/'data').mkdir(exist_ok=True)
    with (ROOT/'data/records.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return rows

def arrays(rows,split):
    chosen=[r for r in rows if r['split']==split]
    return np.array([[r[k] for k in FEATURES] for r in chosen]),np.array([[r[k] for k in LABELS] for r in chosen]),chosen
