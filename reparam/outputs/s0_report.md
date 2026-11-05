# Stage S0: spectrum and unit-scale descriptors

## Models

```
source    cifar10  cifar100  supercifar100  tinyimagenet
family                                                  
ResNet18        6         7              9             6
VGG13           3         3              3             3
ViT             2         2              2             2
```

## Median descriptors per family

```
          bulk_share  bulk_effective_rank  bulk_floor_fraction  unit_scale_cv  bulk_dim
family                                                                                 
ResNet18      0.1256               0.1657               0.0000         0.4260     474.5
VGG13         0.2189               0.1912               0.0212         0.5374     483.5
ViT           0.6849               0.2992               0.0000         0.5162     650.0
```

## Shape distances per source

```
       source  d_vit_vgg  d_resnet_vgg  d_vit_resnet  s1_separates  s2_separates
      cifar10     3.0234        1.4771        4.4783          True          True
     cifar100     0.4300        0.4480        0.1933         False         False
supercifar100     0.8323        0.2975        0.8954          True          True
 tinyimagenet     1.8886        0.5362        2.4241          True          True
```

## Pre-checks

S1 (ViT transport) runs: True

S2 (ResNet-18 sufficiency) runs: True
