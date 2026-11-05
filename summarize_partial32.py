from math import sqrt

full = [0.443,0.428,0.430,0.431,0.391,0.442,0.441,0.446,0.446,0.439,0.435,0.448,0.454,0.431,0.427,0.410,0.441,0.445,0.419,0.431]
no_slow = [0.485,0.460,0.471,0.463,0.460,0.462,0.481,0.499,0.485,0.462,0.482,0.469,0.468,0.470,0.469,0.487,0.500,0.493,0.470]

def stats(xs):
    mean = sum(xs)/len(xs)
    sd = sqrt(sum((x-mean)**2 for x in xs)/(len(xs)-1))
    return mean, sd, sd/sqrt(len(xs))
for name, xs in [('full',full),('no_slow',no_slow)]:
    mean,sd,se=stats(xs)
    print(f'{name}: n={len(xs)} mean={mean:.4f} sd={sd:.4f} se={se:.4f}')
print(f'partial no_slow-full (paired seeds 0-18): mean={sum(b-a for a,b in zip(full[:19],no_slow))/19:.4f}')
