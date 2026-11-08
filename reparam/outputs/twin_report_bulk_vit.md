# Re-origined twin test

## Outcome

**B: origin immaterial**

Primary: B: origin immaterial

```
regime  models  threshold     r  r_twin  delta  ci_low  ci_high
  near      28       0.93 0.448   0.475  0.027  -0.001    0.065
   mid      28       0.88 0.349   0.383  0.033  -0.012    0.094
   far      28       0.50 0.115   0.152  0.036  -0.040    0.122
```

Sensitivity, map-free CSFs from arm none: B: origin immaterial

```
regime  models  threshold     r  r_twin  delta  ci_low  ci_high
  near      28       0.93 0.448   0.475  0.027  -0.001    0.065
   mid      28       0.88 0.349   0.378  0.028  -0.014    0.083
   far      28       0.50 0.115   0.140  0.025  -0.047    0.109
```

## Gates (amendment A1)

Arm bulk_vit.

G1 passed; max gap of MLS, CTM, Confidence to published is 8.66e-05.

G2 passed; max gap between arms is 0.00e+00; ungated Maha gap is 4.58e+00.

G3 passed; max relative NC change is 1.32e-05.

## Rerun noise floor against published AUGRC (descriptive)

```
csf
KPCA RecError global    14.871
Residual                 3.570
PCA RecError global      3.081
Maha                     3.038
NNGuide                  2.405
ViM                      1.952
pNML                     1.314
GEN                      1.149
REN                      1.007
MahaPP                   0.912
fDBD                     0.604
NCI                      0.377
GradNorm                 0.184
NeCo                     0.075
GE                       0.015
Energy                   0.010
PE                       0.008
PCE                      0.007
MSR                      0.007
MLS                      0.000
CTM                      0.000
Confidence               0.000
```

## Rank agreement between arms (descriptive)

```
        spearman  top1_changed  oracle_change
regime                                       
near       0.980         0.135         -0.086
mid        0.978         0.110         -0.056
far        0.973         0.200         -0.135
```

## Mean AUGRC change, bulk_vit minus none (descriptive)

```
regime                near   mid   far
csf                                   
PCA RecError global  -2.15 -4.20 -3.07
pNML                 -1.25 -1.23 -0.98
GradNorm             -1.48 -0.70 -0.04
KPCA RecError global -0.49 -0.43 -0.46
NCI                  -0.19 -0.18 -0.28
fDBD                 -0.16 -0.18 -0.29
NeCo                 -0.07 -0.06 -0.08
NNGuide              -0.08 -0.01 -0.01
REN                   0.00  0.00  0.00
MSR                   0.00  0.00  0.00
MLS                   0.00  0.00  0.00
GEN                   0.00  0.00  0.00
GE                    0.00  0.00  0.00
Energy                0.00  0.00  0.00
Confidence            0.00  0.00  0.00
PCE                   0.00  0.00  0.00
PE                    0.00  0.00  0.00
MahaPP                1.25  0.03  0.28
Residual              2.20  0.06  0.30
ViM                   1.79  0.15  0.19
CTM                  -0.13  0.20  0.15
Maha                  1.20  0.41  0.46
```

## Twin regret of the best 8 CSFs known from arm none (descriptive)

```
regime
near    0.000
mid     0.029
far     0.000
```

## Origin statistics per family (descriptive)

```
          models    dim  zero_fraction  negative_fraction  median_unit_skew  global_mean_norm  class_radius  within_radius  origin_ratio  mean_share_outside_head  bulk_dim  bulk_floor_fraction  bulk_share  bulk_effective_rank  unit_scale_cv
family                                                                                                                                                                                                                                          
ResNet18      28  512.0          0.347              0.000             2.479             4.290         7.662          2.493         0.500                    0.509     474.5                0.000       0.126                0.166          0.426
VGG13         12  512.0          0.307              0.000             1.771             8.184         9.400          6.830         0.865                    0.255     483.5                0.021       0.219                0.191          0.537
ViT            8  768.0          0.000              0.495            -0.001            36.606        65.348         50.984         0.557                    0.974     650.0                0.000       0.685                0.299          0.516
```
