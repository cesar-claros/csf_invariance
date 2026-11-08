# Stage D1: unit-scale tracking test

## Outcome

**D0: the unit scales do not move the rankings enough to test tracking**

## Material change M = regret of the stored shortlist, twin minus original

```
regime  models  threshold     r  r_twin  delta  ci_low  ci_high
  near      28       0.93 0.448   0.320 -0.128  -0.325    0.033
   mid      28       0.88 0.349   0.244 -0.106  -0.305    0.027
   far      28       0.50 0.115   0.147  0.032  -0.063    0.141
```

## Tracking K = new shortlist minus stored shortlist, both on the twin

```
regime  models  threshold     r  r_twin  delta  ci_low  ci_high
  near      28       0.93 0.320   1.378  1.058   0.367    1.922
   mid      28       0.88 0.244   3.454  3.210   1.837    4.813
   far      28       0.50 0.147   3.888  3.741   2.668    4.770
```

## Inputs

Extrapolation share 0.250 (limit 0.25). Mean Jaccard overlap of the shortlists 0.206.

Median change of the z-scored NC inputs:

```
var_collapse           0.006
equiangular_uc         0.108
equiangular_wc         0.195
equinorm_uc            0.514
equinorm_wc            0.533
max_equiangular_uc     0.131
max_equiangular_wc     0.241
self_duality          16.235
```

## Gates

G1 passed (8.66e-05). G2 passed (0.00e+00; ungated Maha 5.71e+00). G5 (amendment A2) passed (head_gap 2.7e-07, stable_gap 0.0162, equinorm_uc_gap 0.132, var_collapse_ratio_over_C_min 0.929, var_collapse_ratio_over_C_max 26.3). G6 passed (max |dp| 0.0e+00). GH passed.
