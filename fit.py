import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# P1 eagles
eagles = pd.DataFrame({
    'y': [17, 29, 17, 20, 1, 15, 0, 1],
    'n': [24, 29, 27, 20, 12, 16, 28, 4],
    'P': [1, 1, 1, 1, 0, 0, 0, 0],
    'A': [1, 1, 0, 0, 1, 1, 0, 0],
    'V': [1, 0, 1, 0, 1, 0, 1, 0]
})
eagles['failures'] = eagles['n'] - eagles['y']

print("P1 Model 1:")
m1 = smf.glm('y + failures ~ P + V + A', data=eagles, family=sm.families.Binomial()).fit()
print(m1.summary())

print("P1 Model 2 (Interaction):")
m2 = smf.glm('y + failures ~ P + V + A + P:A', data=eagles, family=sm.families.Binomial()).fit()
print(m2.summary())

# P2 salamanders
sal = pd.read_csv('exercise/salamanders.csv')
print("P2 Model 1:")
m_sal1 = smf.glm('SALAMAN ~ PCTCOVER', data=sal, family=sm.families.Poisson()).fit()
print(m_sal1.summary())

print("P2 Model 2:")
m_sal2 = smf.glm('SALAMAN ~ PCTCOVER + FORESTAGE', data=sal, family=sm.families.Poisson()).fit()
print(m_sal2.summary())

# P5 Primates
prim = pd.read_csv('exercise/Primates301.csv')
prim = prim.dropna(subset=['social_learning', 'brain', 'research_effort'])
prim['log_brain'] = np.log(prim['brain'])
prim['log_effort'] = np.log(prim['research_effort'])

print("P5 Model 1:")
m_prim1 = smf.glm('social_learning ~ log_brain', data=prim, family=sm.families.Poisson()).fit()
print(m_prim1.summary())

print("P5 Model 2:")
m_prim2 = smf.glm('social_learning ~ log_brain + log_effort', data=prim, family=sm.families.Poisson()).fit()
print(m_prim2.summary())
