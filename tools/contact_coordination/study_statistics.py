"""Paired source-unit counts with explicit unmeasured/unknown outcomes."""
import numpy as np
from scipy.stats import beta,binomtest


def exact_interval(successes,n,alpha=.05):
    if n==0:return None
    return [0. if successes==0 else float(beta.ppf(alpha/2,successes,n-successes+1)),
            1. if successes==n else float(beta.ppf(1-alpha/2,successes+1,n-successes))]


def paired_result(a,b):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    if a.shape!=b.shape or a.ndim!=1:raise ValueError('Matched binary source outcomes required')
    n=len(a)
    if n==0:return dict(paired_N=0,difference_pp=None,McNemar_exact_p=None,paired_interval95_pp=None)
    only_a=int(np.count_nonzero(a&~b));only_b=int(np.count_nonzero(~a&b));discordant=only_a+only_b
    # Simultaneous97.5% binomial intervals cover the two discordant-category
    # probabilities with at least95% joint coverage by Bonferroni. Their
    # difference is an exact conservative interval for the paired effect.
    ia=exact_interval(only_a,n,.025);ib=exact_interval(only_b,n,.025)
    return dict(paired_N=n,A_only=only_a,B_only=only_b,both_success=int(np.count_nonzero(a&b)),neither_success=int(np.count_nonzero(~a&~b)),
        difference_pp=100.*(only_b-only_a)/n,McNemar_exact_p=float(binomtest(only_b,discordant,.5).pvalue) if discordant else 1.,
        paired_interval95_pp=[100*max(-1.,ib[0]-ia[1]),100*min(1.,ib[1]-ia[0])],
        interval_method='Conservative exact paired interval: Bonferroni97.5% Clopper-Pearson intervals for B-only and A-only probabilities, then subtract. Source episode is the paired unit.')


def summarize_policy(rows,source_ids):
    if len(source_ids)!=35 or len(set(source_ids))!=35:raise ValueError('The fixed DEV35 membership is required')
    keyed={}
    for row in rows:
        key=(row['source_id'],row['condition'])
        if key in keyed or key[0] not in source_ids or key[1] not in ('A','B'):raise ValueError('Unexpected/duplicate policy instance')
        keyed[key]=row
    stages=['LEFT_GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','RIGHT_TRANSPORT','BIN_ENTRY','BIN_SETTLE','FULL_TASK']
    conditions={};per_source=[]
    for condition in ('A','B'):
        values=[keyed.get((sid,condition)) for sid in source_ids]
        observed=[v for v in values if v is not None]
        resolved=[v for v in observed if not v.get('unknown',False) and v.get('full_task_success') is not None]
        successes=sum(bool(v['full_task_success']) for v in resolved)
        valid=sum(bool(v.get('valid_physical_rollout',False)) for v in resolved)
        infra=sum(bool(v.get('unknown',False)) for v in observed)
        missing=35-len(observed);unknown=35-len(resolved)
        conditions[condition]=dict(scheduled=35,recorded=len(observed),resolved=len(resolved),not_attempted=missing,infrastructure_unknown=infra,
            unknown_total=unknown,valid_physical_rollouts=valid,full_task_successes=successes,
            primary_status='MEASURED' if unknown==0 else 'NOT_MEASURED' if not observed else 'UNRESOLVED_BOUNDS',
            full_task_rate=successes/35 if unknown==0 else None,full_task_interval95=exact_interval(successes,35) if unknown==0 else None,
            unresolved_rate_bounds=[successes/35,(successes+unknown)/35] if observed and unknown else None,
            conditional_physical_rate=successes/valid if valid else None,
            conditional_physical_status='MEASURED' if valid else 'NOT_MEASURED',
            policy_safety_aborts=sum(v.get('terminal')=='POLICY_SAFETY_ABORT' for v in observed),
            policy_physical_validity_aborts=sum(v.get('terminal')=='POLICY_PHYSICAL_VALIDITY_ABORT' for v in observed),
            cumulative_valid_stage_counts={s:sum(bool(v.get('physical_validity',False) and v.get('stages',{}).get(s,False)) for v in resolved) for s in stages})
        for sid,v in zip(source_ids,values):
            per_source.append(dict(source_id=sid,condition=condition,terminal=v['terminal'] if v else 'NOT_ATTEMPTED_UPSTREAM',
                full_task_success=v.get('full_task_success') if v else None,valid_physical_rollout=bool(v and v.get('valid_physical_rollout')),
                first_failure=v.get('first_failure') if v else 'UPSTREAM_PREREQUISITE',
                **{s:(v.get('stages',{}).get(s,'NOT_ATTEMPTED' if v.get('executed_control_frames')==0 else 'UNKNOWN')
                       if v else 'NOT_ATTEMPTED') for s in stages}))
    all_resolved=all(c['resolved']==35 for c in conditions.values())
    paired=paired_result([keyed[(s,'A')]['full_task_success'] for s in source_ids],[keyed[(s,'B')]['full_task_success'] for s in source_ids]) if all_resolved else dict(paired_N=0,difference_pp=None,McNemar_exact_p=None,paired_interval95_pp=None,status='NOT_ESTIMABLE_UNTIL_MATCHED_OUTCOMES_RESOLVED')
    return dict(conditions=conditions,paired=paired,per_source=per_source,source_units=source_ids)
