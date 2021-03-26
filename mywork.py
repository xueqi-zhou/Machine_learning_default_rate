## data visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date

# encoding='gb18030' deal with Chinese
train_url = '/Users/zhouxueqi/Desktop/PPD_Training_Master_GBK_3_1_Training_Set.csv'
train = pd.read_csv(train_url, encoding='gb18030')
test_url = '/Users/zhouxueqi/Desktop/test.csv'
test = pd.read_csv(test_url, encoding='gb18030')
pd.set_option('display.max_rows',100) #设置最大可见100行
print(train.shape)
combined = pd.concat([train,test])
print(combined.shape)

train_1 = train[train.target==1]
temp = train_1[['ListingInfo','target']].groupby('ListingInfo').agg('sum')*2
temp = temp.rename(columns={'target':'count_1'})
# temp['date'] = temp.index
# temp.date = temp.date.apply(lambda x:(date(int(x.split('-')[0]),int(x.split('-')[1]),int(x.split('-')[2]))-date(2013,11,1)).days)
temp['date'] = pd.to_datetime(temp.index, format='%Y.%m.%d')
temp['days'] = temp.date.dt.day
ax = temp.plot(x='date',y='count_1',title="train set")

train_0 = train[train.target==0]
train_0.target = [1 for _ in range(len(train_0))]
temp_0 = train_0[['ListingInfo','target']].groupby('ListingInfo').agg('sum')
temp_0 = temp_0.rename(columns={'target':'count_0'})
# temp_0['date'] = temp_0.index
# temp_0.date = temp_0.date.apply(lambda x:(date(int(x.split('-')[0]),int(x.split('-')[1]),int(x.split('-')[2]))-date(2013,11,1)).days)
temp_0['date'] = pd.to_datetime(temp_0.index, format='%Y.%m.%d')
temp_0['days'] = temp_0.date.dt.day
temp_0.plot(x='date',y='count_0',ax=ax)

plt.xlabel('Date(20131101~20141109)')
plt.ylabel('count')

plt.legend(['overdue', 'not overdue'])
plt.show()
plt.close()

#--------------------------------------
# data cleaning
#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### plot columns with over 60% missing value ###
def plot_missing_values(df):
    delete_percent = 0.6
    cols = df.columns
    count = [df[col].isnull().sum() for col in cols]  # isnull will generate true,false matrix
    percent = [i / len(df) for i in count]
    missing = pd.DataFrame({'number': count, 'proportion': percent}, index=cols)  # create dataframe
    fig = plt.figure(figsize=(20, 7))

    missing_count = list(zip(cols, count, percent)) #join the data set

    null_col = []
    null_percent = []
    for i in range(len(missing_count)):
        if missing_count[i][1] != 0 and missing_count[i][2] > delete_percent:
            null_col.append(missing_count[i][0])
            null_percent.append(missing_count[i][2])
    plt.bar(null_col, null_percent)
    plt.xlabel('Feature')
    plt.ylabel('Percent')
    title = 'Columns with ' + str(delete_percent * 100) + '% missing value'
    plt.title(title)
    plt.show()
    print(null_col)
    return null_col

if __name__ == '__main__':
    train_delete_missing = plot_missing_values(combined_train_test)


# delete column based on missing value
combined_train_test = combined_train_test.drop(train_delete_missing, axis=1)


### calculate std for every numerical column ###
def standard_d(df):
    cols = df.columns
    delete_std = 0.1
    st_d = []
    for col in cols:
        if df[col].dtypes in ('float64', 'int64'):
            st_d.append(df[col].std())
        else:
            st_d.append('none')
    name_w_std = list(zip(cols,st_d))
    delete_list = []
    for i in range(len(name_w_std)):
        if name_w_std[i][1] != 'none' and name_w_std[i][1] < delete_std:
            delete_list.append(name_w_std[i][0])
    return delete_list


if __name__ == '__main__':
    train_delete_std = standard_d(combined_train_test)

    combine = list(set(train_delete_std))

# delete column based on std
combined_train_test = combined_train_test.drop(combine, axis=1)


### data cleaning on city's name "重庆"="重庆市", keep the first city, remove others 3 ###
def clean_city(df, col):
    for i, name in enumerate(df[col]):
        name = list(name)
        if '市' in name:
            index = name.index('市')
            name = name[0:index]
            df[col].iloc[i] = ''.join(name)

info_city = ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']
for i in info_city:
    clean_city(combined_train_test,i)

#--------------------------------------
# feature engineering
# def function to view overdue rate of different cities
def overdue_rate_city(data, num):
    new = pd.DataFrame({'state/city': data, 'target': combined_train_test.target})
    grouped = new.groupby('state/city')
    count = grouped['target'].agg(np.sum) / grouped['target'].agg(np.size)
    ordered = count.sort_values(ascending=False)
    slicing = ordered[0:num]
    print(slicing)

### see the top 6 overdue rate of citys and state ###
if __name__ == '__main__':
    # num= first num city , means top 6 overdue rate citys
    num = 6
    #overdue rate for every state
    overdue_rate_city(combined_train_test.UserInfo_19, num)
    overdue_rate_city(combined_train_test.UserInfo_7, num)
    #overdue rate for every city
    overdue_rate_city(combined_train_test.UserInfo_2, num)
    overdue_rate_city(combined_train_test.UserInfo_4, num)
    overdue_rate_city(combined_train_test.UserInfo_8, num)
    overdue_rate_city(combined_train_test.UserInfo_20, num)

## overdue rate cutoff = 0.85, we decide to focus on 7 provinces (天津 山东 湖南 辽宁 四川 吉林 海南)
def city_or_not(df, prov_name, col):
    title = prov_name + '_or_not_' + col[-1]
    df[title] = 0
    for i in range(len(df[col])):
        if prov_name in df[col].iloc[i]:
            df[title].iloc[i] = 1
    print(df[title].agg(np.sum)) # how many observation in that prov

if __name__ == '__main__':
    prov_names = ['天津','山东', '湖南', '辽宁', '四川', '吉林', '海南']
    cols = ['UserInfo_7', 'UserInfo_19']
    for i in prov_names:
        for j in cols:
            city_or_not(combined_train_test, i, j)

## test whether the cities are the same in two features
def len_name(col):
    if len(col) == 10:
        return col[-1]
    else:
        return col[-2:]

def test_same(df,col1,col2):
    first_num = len_name(col1)
    second_num = len_name(col2)
    title = 'diff_'+first_num+second_num
    df[title] = 0
    for i in range(len(df[col1])):
        if df[col1].iloc[i] == df[col2].iloc[i]:
            df[title].iloc[i] = 1

data = [combined_train_test]
for i in data:
    test_same(i,'UserInfo_2','UserInfo_4')
    test_same(i,'UserInfo_2','UserInfo_8')
    test_same(i,'UserInfo_2','UserInfo_20')
    test_same(i,'UserInfo_4','UserInfo_8')
    test_same(i,'UserInfo_4','UserInfo_20')
    test_same(i,'UserInfo_8','UserInfo_20')


# give rank of the city instead of name
def rank_city(df, col):
    title = col + '_rank'
    df[title] = 3
    rank_1 = ['北京', '上海', '广州', '深圳']
    rank_2 = ['成都', '杭州', '武汉', '天津', '南京', '重庆', '西安', '长沙', '青岛', '沈阳', '大连', '厦门', '苏州', '宁波', '无锡']
    for i, name in enumerate(df[col]):
        if name in rank_1:
            df[title].iloc[i] = 1
        elif name in rank_2:
            df[title].iloc[i] = 2
    new_num_diff_city = len(np.unique(df[title]))
    print(new_num_diff_city)  # now 395

info_city = ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']
for i in info_city:
    rank_city(combined_train_test, i)


#--------------------------------------
# feature selection
combined_train_test = pd.read_csv('/Users/zhouxueqi/Desktop/combined_train_test.csv')
num_feats = 200
print(combined_train_test.shape)

X = combined_train_test.drop(['target','index'],1)
y = combined_train_test['target']

def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

# chi_selector
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

# rfe_selector
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=200, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

# embeded_lr_selector
embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

# rf_selector
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
# print(str(len(embeded_rf_feature)), 'selected features')

#put all selection together
feature_name = X.columns.tolist()
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support, 'Random Forest': embeded_rf_support})

# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 200
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df) + 1)
# print(feature_selection_df.head(num_feats))
features_by_5models = feature_selection_df['Feature'][:num_feats].tolist()
print(features_by_5models)
print(len(features_by_5models))

feature = combined_train_test.loc[:, features_by_5models]
feature['target'] = combined_train_test['target']

#--------------------------------------
# model
scaler = StandardScaler() # 标准化转换
scaler.fit(X_test)  # 训练标准化对象
x_test_Standard= scaler.transform(X_test)   # 转换数据集
scaler.fit(X_train)  # 训练标准化对象
x_train_Standard= scaler.transform(X_train)   # 转换数据集

#
bp = MLPClassifier(hidden_layer_sizes=(1600,), activation='relu',
                   solver='lbfgs', alpha=0.0001, batch_size='auto',
                   learning_rate='constant')
bp.fit(x_train_Standard, y_train.astype('int'))
y_pred = bp.predict(x_test_Standard)

#--------------------------------------
# Accuracy calculation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

# y_pred_prob = bp.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph")
plt.show()

auc = roc_auc_score(y_test, y_pred_prob)
print(auc)

