# PHydro | Water balanced multitask models
[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### Introduction
Accurate prediction of hydrological variables (HVs) is critical for understanding hydrological processes. Deep learning (DL) models have shown excellent forecasting abilities for different HVs. However, most DL models typically predicted HVs independently, without satisfying the principle of water balance. This missed the interactions between different HVs in the hydrological system and the underlying physical rules. In this study, we developed a DL model based on multitask learning and hybrid physically constrained schemes to simultaneously forecast soil moisture, evapotranspiration, and runoff. The models were trained using ERA5-Land data, which have water budget closure. We thoroughly assessed the advantages of the multitask framework and the proposed constrained schemes. Results showed that multitask models with different loss-weighted strategies produced comparable or better performance compared to the single-task model. The multitask model with a scaling factor of 5 achieved the best among all multitask models and performed better than the single-task model over 70.5% of grids. In addition, the hybrid constrained scheme took advantage of both soft and hard constrained models, providing physically consistent predictions with better model performance. The hybrid constrained models performed the best among different constrained models in terms of both general and extreme performance. Moreover, the hybrid model was affected the least as the training data were arti cially reduced, and provided better spatiotemporal extrapolation ability under different arti cial prediction challenges. These findings suggest that the hybrid model provides better performance compared to previously reported constrained models when facing limited training data and extrapolation challenges.

### Citation

In case you use PHydro in your research or work, please cite:

```bibtex
@article{Lu Li,
    author = {Lu Li, Yongjiu Dai et al.},
    title = {Enforcing Water Balance in Multitask Deep Learning Models for Hydrological Forecasting},
    journal = {Journal of Hydrometeorlogy},
    year = {2024},
    DOI = {10.1175/JHM-D-23-0073.1}
}
```

### [License](https://github.com/leelew/PHydro/LICENSE)
Copyright (c) 2023, Lu Li
