分析泰坦尼克号中性别和仓位等级对存活率的影响，其中性别生成柱形图，仓位等级生成饼图，在一张图中显示

使用的pandas,matplotlib两个工具

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df['Age'] = df['Age'].fillna(0)
df[['Cabin','Embarked']] = df[['Cabin','Embarked']].fillna('null')
df['Sex'] = df['Sex'].map({'male':1, 'female':0})
df.drop_duplicates()

Survived_rate_by_sex = df.groupby('Sex')['Survived'].mean()
Survived_rate_by_Pclass = df.groupby('Pclass')['Survived'].mean()

df = df[df['Age'] != 0]
bins = [0,18,30,40,60,90]
labels = ['0-18','19-30','31-40','41-60','over 61']
df['age_group'] = pd.cut(df['Age'],bins=bins,labels=labels,right=False)
df_copy = df.copy()
df_copy = df_copy[df_copy['Survived'] == 1]
Survived_by_Age_group = df_copy.groupby(['age_group'],observed=False)['Survived'].size()
Survived_Rate_by_Age_group = df.groupby('age_group',observed=False)['Survived'].mean()


plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(2, 2)
ax[0,0].bar(['female','male'], Survived_rate_by_sex.values, width=0.2)
ax[0,0].set_ylim(0,1)
ax[0,0].set_title('Figure 1 : BAR Survived rate by sex')
ax[0,1].pie(Survived_rate_by_Pclass.values, labels=Survived_rate_by_Pclass.index, autopct='%1.1f%%')
ax[0,1].set_title('Figure 2 : PIE Survived rate by Pclass')
ax[1,0].bar(labels, Survived_by_Age_group.values)
ax[1,0].set_title('Figure 3 : Survived by Age group')
ax[1,1].bar(labels, Survived_Rate_by_Age_group.values)
ax[1,1].set_title('Figure 4 : Survived rate by Age group')
plt.tight_layout()
#plt.show()

plt.savefig('ana.png')
