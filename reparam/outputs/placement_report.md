# Step 1: OOD placement and the ViT shortlists

## Outcome

**P0: placement does not carry the shortlist information**

```
comparison   mean    low  high
P - random  0.727  0.073 1.492
     P - B  0.633  0.307 1.005
     P - N  1.085  0.327 2.014
    NP - N -0.163 -0.704 0.438
N - random -0.359 -0.922 0.199
```

## Mean regret per arm, ViT leave one source out

```
          P      N     NP      B  random
all   2.699  1.614  1.451  2.066   1.973
near  1.696  0.754  0.887  1.138   1.475
mid   4.301  2.738  2.560  3.474   2.866
far   0.929  0.593  0.036  0.576   0.895
```

## Mean regret per arm and held-out source (exploratory)

```
                   P      N     NP      B  random
source                                           
cifar10        0.004  0.054  0.006  0.024   0.210
cifar100       0.408  1.124  0.252  0.276   0.945
supercifar100  9.383  5.027  4.933  7.196   5.471
tinyimagenet   0.760  0.057  0.492  0.583   1.164
```

## CNN to ViT: trained on ResNet-18, tested on ViT (descriptive)

```
          P      N     NP      B  random
all   2.718  2.080  1.582  3.769   1.973
near  3.367  2.002  1.757  3.850   1.475
mid   2.461  1.935  1.494  2.800   2.866
far   2.302  2.482  1.507  5.593   0.895
```

## Sensitivity to K (descriptive)

```
 K  P - random  P - B  P - N
10       0.679  0.585  1.299
40       0.401  0.307  0.424
```

## Rows

ViT 310 rows, ResNet-18 218 rows, 21 CSFs. TinyImageNet CIFAR-10 rows excluded.
