from math import sqrt

full = [0.443,0.428,0.430,0.431,0.391,0.442,0.441,0.446,0.446,0.439,0.435,0.448,0.454,0.431,0.427,0.410,0.441,0.445,0.419,0.431]
no_slow = [0.485,0.460,0.471,0.463,0.460,0.462,0.481,0.499,0.485,0.462,0.482,0.469,0.468,0.470,0.469,0.487,0.500,0.493,0.470,0.488]
collapsed = [0.481,0.453,0.460,0.478,0.431,0.456,0.477,0.477,0.484,0.462]

def stats(xs):
    mean=sum(xs)/len(xs); sd=sqrt(sum((x-mean)**2 for x in xs)/(len(xs)-1)); se=sd/sqrt(len(xs))
    return mean,sd,se
for name,xs in [('full',full),('no_slow',no_slow),('collapsed_gamma',collapsed)]:
    mean,sd,se=stats(xs)
    print(f'{name}: n={len(xs)} mean={mean:.4f} sd={sd:.4f} se={se:.4f} ci95=+/-{2.093024*se if len(xs)==20 else 2.262157*se:.4f}')
for name, a,b in [('no_slow-full',no_slow,full),('collapsed-full',collapsed,full[:10]),('collapsed-no_slow',collapsed,no_slow[:10])]:
    ds=[x-y for x,y in zip(a,b)]; mean,sd,se=stats(ds)
    print(f'{name}: n={len(ds)} mean={mean:.4f} ci95=+/-{(2.093024 if len(ds)==20 else 2.262157)*se:.4f} positive={sum(x>0 for x in ds)}/{len(ds)}')
