"""Reproduce README statistics using only the 20 locked test summaries."""
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43, 71, 101)
TASKS = ("gsm8k", "arc", "boolq")


def t4_survival(t):
    """Student t upper tail for four degrees of freedom, by integration."""
    z = t / math.sqrt(4 + t*t)
    return (1-z)**2 * (2+z) / 4


def paired(d):
    assert len(d) == 5
    m, se = mean(d), stdev(d)/math.sqrt(5)
    p = max(t4_survival((m+2)/se), t4_survival((2-m)/se))
    flips = [abs(mean(x*y for x,y in zip(d,s))) for s in itertools.product((-1,1),repeat=5)]
    return dict(differences_pp=d, mean_pp=m, sd_pp=stdev(d),
                ci95=[m-2.7764451052*se,m+2.7764451052*se],
                ci90=[m-2.1318467863*se,m+2.1318467863*se],
                paired_t_p=2*t4_survival(abs(m/se)),
                sign_flip_p=sum(v>=abs(m)-1e-12 for v in flips)/32,
                tost_2pp_p=p, leave_one_out_pp=[mean(d[:i]+d[i+1:]) for i in range(5)])


def holm(ps):
    adjusted = {}
    running = 0
    for i,k in enumerate(sorted(ps,key=ps.get)):
        running = max(running,min(1,(len(ps)-i)*ps[k]))
        adjusted[k] = running
    return adjusted


def main():
    results={}
    for env in ('math','mixed'):
        pairs=[]
        for seed in SEEDS:
            summary=json.loads((ROOT/f'reports/seed{seed}-summary.json').read_text())
            rows={r['operator']:r for r in summary['runs'] if r['environment']==env}
            a,g=rows['am'],rows['gm']
            diffs={k:100*(g['test_metrics'][k]['accuracy']-a['test_metrics'][k]['accuracy']) for k in TASKS}
            primary=diffs['gsm8k'] if env=='math' else sum(w*diffs[k] for w,k in zip((.5,.25,.25),TASKS))
            assert abs(primary-100*(g['primary']-a['primary']))<1e-10
            va,vg=(r['test_metrics']['arc']['valid'] for r in (a,g))
            ca,cg=(r['test_metrics']['arc']['accuracy']/r['test_metrics']['arc']['valid'] for r in (a,g))
            coverage=100*(vg-va)*(cg+ca)/2
            conditional=100*(cg-ca)*(vg+va)/2
            assert abs(coverage+conditional-diffs['arc'])<1e-10
            pairs.append(dict(seed=seed,arc_coverage_pp=coverage,arc_conditional_pp=conditional,am=100*a['primary'],gm=100*g['primary'],primary=primary,**diffs))
        results[env]=dict(pairs=pairs,am=mean(r['am'] for r in pairs),gm=mean(r['gm'] for r in pairs),
                          **paired([r['primary'] for r in pairs]),
                          tasks={k:mean(r[k] for r in pairs) for k in TASKS})
    for test in ('paired_t_p','sign_flip_p','tost_2pp_p'):
        for env,value in holm({e:r[test] for e,r in results.items()}).items():
            results[env][test+'_holm']=value
    results['mixed_minus_math_primary_exploratory']=paired([m['primary']-a['primary'] for m,a in zip(results['mixed']['pairs'],results['math']['pairs'])])
    results['mixed_minus_math_gsm8k_exploratory']=paired([m['gsm8k']-a['gsm8k'] for m,a in zip(results['mixed']['pairs'],results['math']['pairs'])])
    print(json.dumps(results,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
