from enum import unique
import re
import pandas as pd
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import collections




def Data_generation(data):
    data = data.sort_values(['done_time'])
    upper = data.used_time.quantile(0.995)
    data.used_time[data.used_time>=upper]=upper

    ## redefine the output
    time_upper_bound = data.groupby(['question_id'])['used_time'].quantile(0.99)
    time_lower_bound = data.groupby(['question_id'])['used_time'].quantile(0.5) * 0.01
    y = []
    for row in data.itertuples():
        time = row.used_time if row.used_time <= time_upper_bound[row.question_id] else time_upper_bound[row.question_id]
        time = time if time >= time_lower_bound[row.question_id] else time_lower_bound[row.question_id]
        time = (time - time_lower_bound[row.question_id]) / time_upper_bound[row.question_id]
        y.append(time)


    ## feature engineering for the predictor
    data['absolut_used_time'] = data.used_time
    data['used_time'] = y


    X = pd.DataFrame(index=[i for i in range(data.shape[0])])
    ## user feature
    ### 1. user frequency
    n_item_dict = {u:0 for u in data.user_id.unique()}
    n_item = []
    for row in data.itertuples():
        n_item.append(n_item_dict[row.user_id])
        n_item_dict[row.user_id] = n_item_dict[row.user_id] + 1

    X['user_frequency'] = n_item

    ### 2. user adaptive max response time
    global_avg_RT = np.median(data.absolut_used_time)
    max_RT = []
    abs_max_RT = []
    max_RT_dict = {u:0.5 for u in data.user_id.unique()}
    abs_max_RT_dict = {u:global_avg_RT for u in data.user_id.unique()}
    for row in data.itertuples():
        max_RT.append(max_RT_dict[row.user_id])
        abs_max_RT.append(abs_max_RT_dict[row.user_id])
        max_RT_dict[row.user_id] = max(max_RT_dict[row.user_id], row.used_time)
        abs_max_RT_dict[row.user_id] = max(abs_max_RT_dict[row.user_id], row.absolut_used_time)

    X['user_max_RT'] = max_RT
    X['abs_user_max_RT'] = abs_max_RT

    ### 2. user adaptive min response time
    min_RT = []
    abs_min_RT = []
    min_RT_dict = {u:0.5 for u in data.user_id.unique()}
    abs_min_RT_dict = {u:global_avg_RT for u in data.user_id.unique()}
    for row in data.itertuples():
        min_RT.append(min_RT_dict[row.user_id])
        abs_min_RT.append(abs_min_RT_dict[row.user_id])
        min_RT_dict[row.user_id] = min(min_RT_dict[row.user_id], row.used_time)
        abs_min_RT_dict[row.user_id] = min(abs_min_RT_dict[row.user_id], row.absolut_used_time)
    X['user_min_RT'] = min_RT
    X['abs_user_min_RT'] = abs_min_RT


    ### 3. user adaptive average item response time
    average_RT = []
    abs_average_RT = []
    average_RT_dict = {u:[0.5] for u in data.user_id.unique()}
    abs_average_RT_dict = {u:[global_avg_RT] for u in data.user_id.unique()}
    for row in data.itertuples():
        average_RT.append(np.mean(average_RT_dict[row.user_id][:]))
        abs_average_RT.append(np.mean(abs_average_RT_dict[row.user_id][:]))
        average_RT_dict[row.user_id].append(row.used_time)
        abs_average_RT_dict[row.user_id].append(row.absolut_used_time)
    X['user_avg_RT'] = average_RT
    X['abs_user_avg_RT'] = abs_average_RT

    ### 4. user adaptive average last 3 response time
    average_last_3_RT = []
    abs_average_last_3_RT = []
    average_last_3_RT_dict = {u:[0.5] for u in data.user_id.unique()}
    abs_average_last_3_RT_dict = {u:[global_avg_RT] for u in data.user_id.unique()}
    for row in data.itertuples():
        value = average_last_3_RT_dict[row.user_id]
        abs_value = abs_average_last_3_RT_dict[row.user_id]
        if len(value) <= 3:
            value = np.mean(value)
            abs_value = np.mean(abs_value)
        else:
            value = np.mean(value[-3:])
            abs_value = np.mean(abs_value[-3:])
        average_last_3_RT.append(value)
        abs_average_last_3_RT.append(abs_value)
        average_last_3_RT_dict[row.user_id].append(row.used_time)
        abs_average_last_3_RT_dict[row.user_id].append(row.absolut_used_time)
    X['user_avgLast3_RT'] = average_last_3_RT
    X['abs_user_avgLast3_RT'] = abs_average_last_3_RT


    ### last 5 item accuracy
    average_last_5_Acc = []
    average_last_5_Acc_dict = {u:[0.5] for u in data.user_id.unique()}
    for row in data.itertuples():
        value = average_last_5_Acc_dict[row.user_id]
        if len(value) <= 5:
            value = np.mean(value)
        else:
            value = np.mean(value[-5:])
        average_last_5_Acc.append(value)
        average_last_5_Acc_dict[row.user_id].append(row.if_answered_correct)
    X['average_last_5_Acc'] = average_last_5_Acc
    

    ### last date accuracy / counts / average_RT
    data['start_date'] = pd.to_datetime(data['start_date'])
    data['start_date'] = data.groupby(['user_id'])['start_date'].rank(method='min')
    user_date = {u:[] for u in data.user_id.unique()}
    for row in data.itertuples():
        user_date[row.user_id].append(row.start_date)
    user_date_count_dict = {}
    for i,j in user_date.items():
        user_date_count_dict[i] = dict(collections.Counter(j))


    user_date_new = {}
    for i,j in user_date.items():
        user_date_new[i] = sorted(list(set(j)))


    user_date_acc_dict = data.groupby(['user_id','start_date'])['if_answered_correct'].mean()
    user_date_avg_abs_RT_dict = data.groupby(['user_id','start_date'])['absolut_used_time'].mean()
    user_date_avg_RT_dict = data.groupby(['user_id','start_date'])['used_time'].mean()

    user_date_count = []
    user_date_acc = []
    user_date_avg_abs_RT = []
    user_date_avg_RT = []
    for row in data.itertuples():
        if row.start_date == 1:
            user_date_acc.append(0.5)
            user_date_avg_abs_RT.append(global_avg_RT)
            user_date_avg_RT.append(0.5)
            user_date_count.append(0)
        else:
            index1 = row.user_id
            index2 = user_date_new[row.user_id][user_date_new[row.user_id].index(row.start_date)-1]
            user_date_acc.append(user_date_acc_dict[index1,index2])
            user_date_avg_abs_RT.append(user_date_avg_abs_RT_dict[index1,index2])
            user_date_avg_RT.append(user_date_avg_RT_dict[index1,index2])
            user_date_count.append(user_date_count_dict[index1][index2])

    X['user_date_count'] = user_date_count
    X['user_date_acc'] = user_date_acc
    X['user_date_avg_abs_RT'] = user_date_avg_abs_RT
    X['user_date_avg_RT'] = user_date_avg_RT



    ## item feature
    ### 1. adaptive item difficulity
    item_diff = []
    item_diff_global = 1 - np.mean(data.if_answered_correct)
    item_diff_dict = {i:[item_diff_global] for i in data.question_id.unique()}
    for row in data.itertuples():
        value = item_diff_dict[row.question_id]
        item_diff.append((5*item_diff_global+np.sum(value))/(5+len(value)))
        item_diff_dict[row.question_id].append(1-row.if_answered_correct)
    X['item_diff'] = item_diff    


    ### 2. item time intensity
    item_time = []
    item_time_global = 0.5
    item_time_dict = {i:[item_time_global] for i in data.question_id.unique()}
    for row in data.itertuples():
        value = item_time_dict[row.question_id]
        item_time.append((5*item_time_global+np.sum(value))/(5+len(value)))
        item_time_dict[row.question_id].append(row.used_time)
    X['item_time'] = item_time    

    ### 3. item frequency
    n_user_dict = {i:0 for i in data.question_id.unique()}
    n_question = []
    for row in data.itertuples():
        n_question.append(n_user_dict[row.question_id])
        n_user_dict[row.question_id] = n_user_dict[row.question_id] + 1
    X['item_frequency'] = n_question


    #### 4. item lesson
    lessons = pd.get_dummies(data.lesson_id, prefix='lesson')
    X = pd.concat([X,lessons],axis=1)


    #### 5. item chapter
    chapter = pd.get_dummies(data.chapter, prefix='chapter')
    X = pd.concat([X,chapter],axis=1)


    #### 6. if anchor 
    is_anchor_question = data.is_anchor_question
    X['is_anchor_question'] = is_anchor_question

    #### 7. expert-defined difficulity
    difficulty = data.difficulty
    X['difficulty'] = difficulty

    #### 8. knowledege informaiton
    column0 = ['level_4_knowledge_id', 'level_3_knowledge_id', 'level_2_knowledge_id','level_1_knowledge_id']
    def change(x):
        return ','.join(re.findall('[^0-9](130[0-9]+)', x))
    for i in column0:
        data[i]=data[i].apply(lambda x: change(x))

    level_1 = pd.get_dummies(data.level_1_knowledge_id,prefix='level_1')
    X = pd.concat([X,level_1],axis=1)
    level_2 = pd.get_dummies(data.level_2_knowledge_id,prefix='level_2')
    X = pd.concat([X,level_2],axis=1)
    level_3 = pd.get_dummies(data.level_3_knowledge_id,prefix='level_3')
    X = pd.concat([X,level_3],axis=1)
    level_4 = pd.get_dummies(data.level_4_knowledge_id,prefix='level_4')
    X = pd.concat([X,level_4],axis=1)

    
    #### 9. last 5 item average diff
    average_last_5_diff = []
    item_diff_global = 1 - np.mean(data.if_answered_correct)
    average_last_5_diff_dict = {i:[item_diff_global] for i in data.question_id.unique()}
    for row in data.itertuples():
        value = average_last_5_diff_dict[row.question_id]
        if len(value) <= 5:
            value = np.mean(value)
        else:
            value = np.mean(value[-5:])
        average_last_5_diff.append(value)
        average_last_5_diff_dict[row.question_id].append(1-row.if_answered_correct)
    X['average_last_5_diff'] = average_last_5_diff

    # interaction feature
    ### item position in a lesson
    X['item_position_lesson'] = data.groupby(['user_id','lesson_id'])['done_time'].rank(ascending=True)
    X['item_poisition_day'] = data.groupby(['user_id','start_date'])['done_time'].rank(ascending=True)
    
    ## other parameters
    if_first_question = data.if_first_question
    X['if_first_question']=if_first_question
    is_fusing = data.is_fusing
    X['is_fusing']=is_fusing
    if_finish_lesson = data.if_finish_lesson
    X['if_finish_lesson']=if_finish_lesson

    X['y'] = y
    X['abs_y'] = data['absolut_used_time']
    
    X.to_csv('./data/data.csv')





