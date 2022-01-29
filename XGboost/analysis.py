import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

raw = pd.read_csv('./data/inputdata.csv').drop(columns=['Unnamed: 0'])
np.sum(raw.done_time.isna())
raw['done_time'] = raw['done_time'].astype(int) 
raw = raw[raw.used_time<=300]


plt.hist(raw[raw.chapter=='集合'].used_time,range=[0, 300],bins=100)
plt.title('jihe')
plt.show()


plt.hist(raw[raw.if_answered_correct==1].used_time,range=[0, 300],bins=100)
plt.title('correct')
plt.show()





raw['item_position_lesson'] = raw.groupby(['user_id','lesson_id'])['done_time'].rank(ascending=True,method='min')
sns.boxplot(x='item_position_lesson', y='used_time', data=raw)
plt.show()


raw['item_poisition_day'] = raw.groupby(['user_id','start_date'])['done_time'].rank(ascending=True,method='min')
sns.boxplot(x='item_poisition_day', y='used_time', data=raw)
plt.show()


import re
column0 = ['level_4_knowledge_id', 'level_3_knowledge_id', 'level_2_knowledge_id','level_1_knowledge_id']
def change(x):
    return ','.join(re.findall('[^0-9](130[0-9]+)', x))
for i in column0:
    raw[i]=raw[i].apply(lambda x: change(x))


sns.boxplot(x='level_1_knowledge_id', y='used_time', data=raw)
plt.show()

sns.boxplot(x='level_2_knowledge_id', y='used_time', data=raw)
plt.show()

sns.boxplot(x='level_3_knowledge_id', y='used_time', data=raw)
plt.show()

sns.boxplot(x='level_4_knowledge_id', y='used_time', data=raw)
plt.show()

# np.min(raw.start_date)
# np.max(raw.start_date)


plt.hist(raw.groupby('question_id')['used_time'].median())
plt.show()
