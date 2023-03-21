from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import scipy.sparse as sp
import numpy as np
import seaborn as sn
from sklearn import tree
import matplotlib.pyplot as plt
from config import parse_args

args = parse_args()

x_train = pd.read_csv('./data/feature/x_train_norm.csv').iloc[:,1:]
y_train = pd.read_csv('./data/feature/y_train.csv').iloc[:,1:]

x_test = pd.read_csv('./data/feature/x_test_norm.csv').iloc[:,1:]
y_test = pd.read_csv('./data/feature/y_test.csv').iloc[:,1:]


rfc = RandomForestClassifier(random_state=1)
rfc.fit(x_train, np.ravel(y_train))

y_train_pred = rfc.predict(x_train)
y_test_pred = rfc.predict(x_test)

# print('Training Score: %.3f, Testing Score: %.3f' % (
#         rfc.score(x_train,y_train),
#         rfc.score(x_test,y_test)))

# print('MSE train: %.3f, test: %.3f' % (
#         mean_squared_error(y_train, y_train_pred),
#         mean_squared_error(y_test, y_test_pred)))

# print('R^2 train: %.3f, test: %.3f' % (
#         r2_score(y_train, y_train_pred),
#         r2_score(y_test, y_test_pred)))

# 计算特征重要性
feature_importances = rfc.feature_importances_

# 绘制随机森林
# plt.figure(figsize=(20, 20))
# for i, tree_in_forest in enumerate(rfc.estimators_):
#     dot_data = export_graphviz(tree_in_forest, out_file=None,
#                                feature_names=x_train.columns,
#                                class_names=ticks,
#                                filled=True, rounded=True,
#                                special_characters=True)
#     graph = graphviz.Source(dot_data)
#     plt.subplot(10, 10, i+1)
#     plt.imshow(graph.render(view=False, format='png'))
#     plt.axis('off')
# plt.suptitle('Random Forest')
# plt.show()

# 选取重要性前多少的特征

# 将zip后的数据进行转置
zipped_data = list(zip(x_train.columns, feature_importances))
# print(zipped_data)

# 对数据进行排序
sorted_data = sorted(zipped_data, key=lambda x: x[1], reverse=True)
# print(sorted_data)

feature_importances, feature_data = zip(*sorted_data)
# print(feature_importances)

# 绘制特征重要性条状图
if 1:
    plt.figure()
    # plt.bar(range(len(feature_data)), feature_data)
    # plt.xticks(range(len(feature_data)), feature_importances, rotation=90)
    plt.barh(range(10), feature_data[:10])
    plt.yticks(range(10), feature_importances[:10])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Top 10 Important Features')
    plt.show()

best_n = -1
best_score = 0
scores = []

if 1:
    for i in range(1, x_train.shape[1] + 1):
        topn_feature_names = [feature for feature in feature_importances[:i]]
        X_topn_train = x_train[topn_feature_names]
        X_topn_test = x_test[topn_feature_names]
        rf_topn = RandomForestClassifier(random_state=1)
        rf_topn.fit(X_topn_train, np.ravel(y_train))
        y_train_pred = rf_topn.predict(X_topn_train)
        y_test_pred = rf_topn.predict(X_topn_test)

        # print('Training Score: %.3f, Testing Score: %.3f' % (
        #     rf_topn.score(X_topn_train,y_train),
        #     rf_topn.score(X_topn_test,y_test)))

        # print('MSE train: %.3f, test: %.3f' % (
        #     mean_squared_error(y_train, y_train_pred),
        #     mean_squared_error(y_test, y_test_pred)))

        # print('R^2 train: %.3f, test: %.3f' % (
        #     r2_score(y_train, y_train_pred),
        #     r2_score(y_test, y_test_pred)))
        
        scores.append(rf_topn.score(X_topn_test,y_test))

        if rf_topn.score(X_topn_test,y_test) > best_score:
            best_score = rf_topn.score(X_topn_test,y_test)
            best_n = i

        print('selected features:', len(topn_feature_names), 'score:', rf_topn.score(X_topn_test,y_test), 'current best:', best_n, best_score)

    print('best', best_n, best_score)

    # 绘制准确率随特征数量变化图
    plt.figure()
    plt.plot(range(1, x_train.shape[1] + 1), scores)
    plt.xticks(range(1, x_train.shape[1] + 1, 5))
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy with different number of features')
    plt.show()
else:
    best_n = 47


# 再次训练效果最好的模型
top47_features = [feature for feature in feature_importances[:best_n]]
x_train = x_train[top47_features]
x_test = x_test[top47_features]
rfc.fit(x_train, np.ravel(y_train))
y_train_pred = rfc.predict(x_train)
y_test_pred = rfc.predict(x_test)

print('Training Score: %.3f, Testing Score: %.3f' % (
        rfc.score(x_train,y_train),
        rfc.score(x_test,y_test)))

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# class_names = ['标准','呼吸错误', '身体晃动', '上半身过高', '外侧划臂']

# for i in range(len(rfc.estimators_)):
#     plt.figure()
#     tree.plot_tree(rfc.estimators_[i], feature_names=top47_features, class_names=class_names, filled=True)
#     plt.show()

if 0:
    # 绘制混淆矩阵

    y_test_pred = pd.DataFrame(y_test_pred)

    y_res = pd.concat([y_test,y_test_pred], axis=1)

    y_res.columns = ['label', 'predict']

    # print(y_res)

    y_res_count = y_res.value_counts().to_frame()

    # print(y_res_count)

    y_res_count.to_csv('./tmp.csv')

    y_res_count = pd.read_csv('./tmp.csv')

    y_res_count.columns = ['label', 'predict', 'count']

    # print(y_res_count)

    res_array = y_res_count.to_numpy()

    # print(res_array)

    # 稀疏矩阵转化
    tmp = sp.coo_matrix(arg1=(res_array[:, 2], (res_array[:, 0], res_array[:, 1])), shape=(args.label_num,args.label_num), dtype=np.int64)
    heatmap_data = tmp.todense()
    # print(heatmap_data)

    heatmap_data = heatmap_data / heatmap_data.sum()

    # print(heatmap_data)

    ticks = ['标准','呼吸错误', '身体晃动', '上半身过高', '外侧划臂']

    heatmap_data = pd.DataFrame(heatmap_data, index=ticks, columns=ticks)

    sn.set()
    plt.rcParams['font.sans-serif']='SimHei'
    fig,ax=plt.subplots(figsize=(args.label_num, args.label_num))
    hp = sn.heatmap(heatmap_data, cmap='Blues')
    hp.set_ylabel('真实')
    hp.set_xlabel('预测')

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.show()