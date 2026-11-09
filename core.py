"""Explicit per-label decision thresholds and portable learned linear weights."""
import numpy as np

FEATURES=['amount_ratio','event_gap','document_count','source_agreement','repeat_count','channel_score','distance_signal','tenure_signal']
LABELS=['document_gap','timing_review','amount_conflict','specialist_route']

def check_matrix(value,columns,name):
    a=np.asarray(value,dtype=float)
    if a.ndim!=2 or a.shape[1]!=columns or len(a)==0 or not np.isfinite(a).all():raise ValueError(name+' shape or finite-value violation')
    return a

def probabilities(x,artifact):
    x=check_matrix(x,len(FEATURES),'features')
    mean=np.asarray(artifact['mean']);scale=np.asarray(artifact['scale']);coef=np.asarray(artifact['coef']);intercept=np.asarray(artifact['intercept'])
    if artifact['features']!=FEATURES or artifact['labels']!=LABELS:raise ValueError('artifact feature/label order mismatch')
    if mean.shape!=(len(FEATURES),) or scale.shape!=mean.shape or coef.shape!=(len(LABELS),len(FEATURES)) or intercept.shape!=(len(LABELS),):raise ValueError('artifact dimensions')
    if not all(np.isfinite(a).all() for a in [mean,scale,coef,intercept]) or (scale<=0).any():raise ValueError('invalid artifact values')
    logits=(x-mean)/scale@coef.T+intercept
    return 1/(1+np.exp(-np.clip(logits,-40,40)))

def binary_f1(y,pred):
    y=np.asarray(y,dtype=bool);pred=np.asarray(pred,dtype=bool)
    tp=np.sum(y&pred,axis=0);fp=np.sum(~y&pred,axis=0);fn=np.sum(y&~pred,axis=0)
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)

def choose_thresholds(y,p,min_support=25):
    p=check_matrix(p,len(LABELS),'probabilities');y=check_matrix(y,len(LABELS),'labels')
    if p.shape!=y.shape or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():raise ValueError('invalid paired labels/probabilities')
    if isinstance(min_support,bool) or not isinstance(min_support,int) or min_support<1:raise ValueError('positive support count required')
    thresholds=[];audit=[]
    for j,label in enumerate(LABELS):
        positive=int(y[:,j].sum());negative=len(y)-positive
        if min(positive,negative)<min_support:
            thresholds.append(None);audit.append({'label':label,'status':'withheld','positive':positive,'negative':negative,'reason':'insufficient_validation_support'});continue
        candidates=[.05,.10,.15,.20,.25,.30,.40,.50,.60,.70,.80]
        chosen=max(candidates,key=lambda t:(float(binary_f1(y[:,j],p[:,j]>=t)),t))
        thresholds.append(chosen);audit.append({'label':label,'status':'supported','positive':positive,'negative':negative,'threshold':chosen})
    return thresholds,audit

def decisions(probability,thresholds):
    p=check_matrix(probability,len(LABELS),'probabilities')
    if ((p<0)|(p>1)).any():raise ValueError('probabilities must lie in [0,1]')
    if len(thresholds)!=len(LABELS):raise ValueError('threshold count')
    result=np.full(p.shape,-1,dtype=int)
    for j,t in enumerate(thresholds):
        if t is None:continue
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not np.isfinite(t) or not 0<=t<=1:raise ValueError('threshold must be finite probability or None')
        result[:,j]=(p[:,j]>=t).astype(int)
    return result
