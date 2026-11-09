# Rerun variability and the ViT failure (exploratory)

## Checks

ViT paper: predictor regret matches the pipeline on 320 rows (max deviation 0.0e+00)

ViT paper: random regret matches the pipeline on 320 rows (max deviation 6.5e-14)

ViT rerun: predictor regret matches the pipeline on 320 rows (max deviation 0.0e+00)

ViT rerun: random regret matches the pipeline on 320 rows (max deviation 6.9e-14)

ResNet18 paper: predictor regret matches the pipeline on 224 rows (max deviation 0.0e+00)

ResNet18 rerun: predictor regret matches the pipeline on 224 rows (max deviation 0.0e+00)

## ViT: headroom H = (q - o) / q

```
regime  rows  models  predictor_r  random_q  cross_oracle_o     H  H_low  H_high                              reading  spearman  same_best
  near   110      40        1.875     1.365           0.365 0.733  0.526   0.893                              partial     0.638      0.509
   mid   140      40        3.907     2.979           0.541 0.818  0.663   0.934                              partial     0.807      0.443
   far    70      40        3.295     0.957           0.024 0.975  0.931   0.996 noise explains little of the failure     0.915      0.800
```

## ResNet18: headroom H = (q - o) / q

```
regime  rows  models  predictor_r  random_q  cross_oracle_o     H  H_low  H_high                              reading  spearman  same_best
  near    74      28        0.496     1.482           0.007 0.995  0.986     1.0 noise explains little of the failure     0.995      0.919
   mid   100      28        0.378     1.156           0.000 1.000  1.000     1.0 noise explains little of the failure     0.994      0.960
   far    50      28        0.112     1.225           0.004 0.997  0.991     1.0 noise explains little of the failure     0.995      0.940
```
