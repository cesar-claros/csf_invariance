# Re-origined twin test

## Outcome

**B: origin immaterial**

Primary: B: origin immaterial

```
regime  models  threshold     r  r_twin  delta  ci_low  ci_high
  near      28       0.93 0.448   0.490  0.042  -0.232    0.309
   mid      28       0.88 0.349   0.592  0.242   0.125    0.377
   far      28       0.50 0.115   0.223  0.108   0.020    0.202
```

Sensitivity, origin-free CSFs from arm none: B: origin immaterial

```
regime  models  threshold     r  r_twin  delta  ci_low  ci_high
  near      28       0.93 0.448   0.490  0.042  -0.232    0.309
   mid      28       0.88 0.349   0.592  0.242   0.125    0.377
   far      28       0.50 0.115   0.223  0.108   0.020    0.202
```

## Gates (amendment A1)

G1 passed; max gap of MLS, CTM, Confidence to published is 8.66e-05.

G2 passed; max gap between arms is 4.84e-05; ungated Maha gap is 1.03e-01.

G3 passed; max relative NC change is 3.43e-05.

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
near       0.912         0.514          0.302
mid        0.867         0.350          0.054
far        0.925         0.380          0.053
```

## Mean AUGRC change, center minus none (descriptive)

```
regime                 near    mid    far
csf                                      
GradNorm             -35.61 -39.95 -37.58
PCA RecError global  -15.10  -6.77  -5.26
pNML                 -11.11  -4.18  -2.95
MahaPP                -9.29  -2.38  -1.60
NCI                   -1.49  -0.84  -0.26
Residual               0.00  -0.09  -0.04
ViM                    0.01  -0.03  -0.02
GEN                    0.00   0.00   0.00
Energy                 0.00   0.00   0.00
MLS                    0.00   0.00   0.00
MSR                    0.00   0.00   0.00
Maha                  -0.00  -0.00   0.00
fDBD                  -0.00   0.00  -0.00
GE                     0.00   0.00   0.00
NeCo                   0.00   0.00   0.00
Confidence             0.00   0.00   0.00
PCE                    0.00   0.00   0.00
PE                     0.00   0.00   0.00
REN                    0.00   0.00   0.00
NNGuide               -0.36   0.10   0.05
KPCA RecError global  -0.58   0.50   0.76
CTM                    1.31   2.80   2.16
```

## Twin regret of the best 8 CSFs known from arm none (descriptive)

```
regime
near    0.052
mid     0.107
far     0.000
```

## Origin statistics per family (descriptive)

```
          models    dim  zero_fraction  negative_fraction  median_unit_skew  global_mean_norm  class_radius  within_radius  origin_ratio  mean_share_outside_head
family                                                                                                                                                           
ResNet18      28  512.0          0.347              0.000             2.479             4.290         7.662          2.493         0.500                    0.509
VGG13         12  512.0          0.307              0.000             1.769             8.184         9.391          6.824         0.865                    0.255
ViT            8  768.0          0.000              0.495            -0.001            36.606        65.348         50.984         0.557                    0.974
```
