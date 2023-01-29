import autograd.numpy as np 

import ssm_star

###
# Configs 
###

K_true = 2 
D_true = 3
N =  4

###
# Sampling
###
print("Sampling SLDS with system-driven transitions")
slds = ssm_star.SLDS(N, K_true, D_true, emissions="gaussian", transitions="system_driven")

# insight into system level regimes 
print(f"Xi, the KxL t.p.m governing how current system regime influences current entity regime is: \n {slds.transitions.Xis}")
print(f"lambda_, the strength of influence of system level regimes on transitions between entity regimes is: {slds.transitions.lambda_}")

z,x,y=slds.sample(T=100)
