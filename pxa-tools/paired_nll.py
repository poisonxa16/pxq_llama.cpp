#!/usr/bin/env python3
"""Paired per-chunk NLL analysis. Usage: paired_nll.py <dir> <tagA> <tagB> [...more tags vs tagA]
Reads <dir>/nll-<tag>.txt series (one chunk NLL per line), reports marginal ppl with honest
chunk-level bars and the paired delta vs tagA."""
import sys, math

d = sys.argv[1]
tags = sys.argv[2:]
series = {}
for t in tags:
    series[t] = [float(x) for x in open("%s/nll-%s.txt" % (d, t))]
n = min(len(v) for v in series.values())
print("chunks used: %d" % n)
for t in tags:
    v = series[t][:n]
    m = sum(v)/n
    sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1))
    print("%-24s ppl=%.4f  mean_nll=%.6f  chunk_sd=%.4f  se=%.5f" % (t, math.exp(m), m, sd, sd/math.sqrt(n)))
a = tags[0]
for t in tags[1:]:
    dv = [series[t][i] - series[a][i] for i in range(n)]
    m = sum(dv)/n
    sd = math.sqrt(sum((x-m)**2 for x in dv)/(n-1))
    se = sd/math.sqrt(n)
    tstat = m/se if se > 0 else float("inf")
    print("PAIRED %s - %s: delta=%.6f nats  se=%.6f  t=%.2f  (%.3f%% ppl)"
          % (t, a, m, se, tstat, 100*(math.exp(m)-1)))
