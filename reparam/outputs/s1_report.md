# Stage S1: ViT transport test

## Outcome

**Z: inconclusive**

Primary: Z: inconclusive

```
regime  rows  models   gap  gap_twin      e  e_low  e_high
  near   110      40 0.873     1.030 -0.157 -0.654   0.308
   mid   140      40 1.043    -0.106  1.149  0.331   2.091
   far    70      40 2.278     2.503 -0.225 -0.630   0.253
```

## Control arm bulk_away

not run

## Gates

Arm bulk_vgg, 40 models.

G1 (amendment A3) passed; median gap of MLS, CTM to published is 5.92e-04 (largest single gap 5.59).

G2 passed; max gap between arms is 0.00e+00; ungated Maha gap is 7.71e+00.

G3 passed; max relative NC change is 1.27e-06.

G4 passed; the rebuilt ViT shortlists reproduce the stored per-row regret (max deviation 3.6e-15).

## Per source (exploratory)

```
       family regime  rows  models    gap  gap_twin      e  e_low  e_high
     cifar100   near    20      10  0.075    -0.019  0.094  0.044   0.146
     cifar100    mid    40      10  0.181     0.110  0.071 -0.122   0.212
     cifar100    far    20      10  0.368     0.177  0.192  0.075   0.369
      cifar10   near    20      10  0.012     0.654 -0.642 -1.396   0.090
      cifar10    mid    40      10  0.065     0.018  0.047 -0.076   0.154
      cifar10    far    20      10 -0.027     0.050 -0.077 -0.250   0.094
supercifar100   near    20      10  3.826     5.194 -1.368 -3.239   0.819
supercifar100    mid    40      10  1.980    -2.088  4.068  1.996   6.249
supercifar100    far    20      10  7.912     8.798 -0.887 -2.039   0.748
 tinyimagenet   near    50      10  0.357    -0.064  0.421  0.044   0.834
 tinyimagenet    mid    20      10  2.853     3.180 -0.327 -1.205   0.636
 tinyimagenet    far    10      10 -0.562    -0.531 -0.031 -0.115   0.064
```

## Rerun noise floor against published AUGRC (descriptive)

```
csf
REN                     222.380
pNML                    196.991
MahaPP                  186.508
Maha                    186.346
Residual                166.047
PCA RecError global     138.421
ViM                      62.247
KPCA RecError global     62.122
NNGuide                  30.507
fDBD                     18.574
NCI                      10.047
MLS                       5.594
NeCo                      5.458
Energy                    5.445
CTM                       5.444
GradNorm                  5.425
MSR                       5.085
GEN                       5.014
PCE                       4.978
GE                        4.924
PE                        4.832
```
