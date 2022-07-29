# cpasocp
Chambolle-Pock Algorithm Solving Optimal Control Problems

## Overview
Implement the Chambolle-Pock algorithm in Python and compare it to state-of-the-art algorithms (e.g. ADMM).

Furthermore, using acceleration algorithms to speed it up.
<br />

### Implemented algorithms
- Chambolle-Pock method
- Alternating Direction Method of Multipliers
- Chambolle-Pock method with scaling constraints
- ADMM with scaling constraints
- SuperMann acceleration
- Anderson's acceleration (for choosing direction for SuperMann)
<br />
<br />

## Results
### Chambolle-Pock method Residuals vs Iterations
![comparison all](./results/Residual_semilogy.jpg)
<br />
<br />

### Chambolle-Pock method with SuperMann Residuals vs Iterations
![comparison all](./results/SuperMann_Residual_semilogy.jpg)
<br />
<br />

### Comparison all performance profile
![comparison all](./results/comparison%20all/comparison_semilog.jpg)
<br />
<br />

### Comparison CP ADMM performance profile
![comparison CP ADMM](./results/comparison%20CP%20ADMM/comparison_semilog.jpg)
<br />
<br />

### Comparison CP ADMM scaling performance profile
![comparison CP ADMM scaling](./results/comparison%20CP%20scaling/comparison_semilog.jpg)
<br />
<br />

### Comparison scaling performance profile
![comparison scaling](./results/comparison%20scaling/comparison_semilog.jpg)
<br />
<br />
