import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

df = df.drop_duplicates()
df['Age'] = df['Age'].fillna('0')
df['Cabin']  = df['Cabin'].fillna('null')
df['Embarked']  = df['Embarked'].fillna('null')
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

survival_rate_by_sex = df.groupby('Sex')['Survived'].mean()
survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean()

fig, ax = plt.subplots(2,1)
ax[0].bar(['male','female'], survival_rate_by_sex.values)
ax[0].set_title('Survival Rate by Gender', fontsize=14)
ax[0].set_xlabel('Sex',)
ax[0].set_ylabel('Rate')
ax[0].set_ylim(0,1)
ax[1].pie(survival_rate_by_pclass.values, labels=survival_rate_by_pclass.index, autopct='%1.1f%%')
ax[1].set_title('Survival Rate by Pclass')
plt.tight_layout()
plt.show()