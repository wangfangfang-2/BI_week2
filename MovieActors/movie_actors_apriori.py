# 分析MovieLens 电影分类中的频繁项集和关联规则
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 数据加载
movies = pd.read_csv('./movie_actors.csv')
#print(movies.head())
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
movies_hot_encoded = movies.drop('actors',1).join(movies.actors.str.get_dummies('/'))
pd.options.display.max_columns=100
print(movies_hot_encoded.head())

# 将movieId, title设置为index
movies_hot_encoded.set_index(['title'],inplace=True)
#print(movies_hot_encoded.head())
# 挖掘频繁项集，最小支持度为0.02
itemsets = apriori(movies_hot_encoded,use_colnames=True, min_support=0.05)
# 按照支持度从大到小进行时候粗
itemsets = itemsets.sort_values(by="support" , ascending=False) 
print('-'*20, '频繁项集', '-'*20)
print(itemsets)
# 根据频繁项集计算关联规则，设置最小提升度为2
rules =  association_rules(itemsets, metric='lift', min_threshold=2)
# 按照提升度从大到小进行排序
rules = rules.sort_values(by="lift" , ascending=False) 
#rules.to_csv('./rules.csv')
print('-'*20, '关联规则', '-'*20)
print(rules)

#以下为结果部分
            title  GangZhao  LukeZhiGangLiu  PengZhenZhong  YuanFang  一纳  丁嘉丽  \
0      囧妈‎ (2020)         0               0              0         0   0    0   
1  我和我的祖国‎ (2019)         0               0              0         0   0    0   
2   我不是药神‎ (2018)         0               0              0         0   0    0   
3  疯狂的外星人‎ (2019)         0               0              0         0   0    0   
4   疯狂的石头‎ (2006)         0               0              0         0   0    0   

   丁志城  丁志诚  丁黑  万弘杰  严敏  严晓频  中孝介  丹尼尔·海尼  乔任梁  九孔  于和伟  于波  于荣光  于谦  仁龙  \
0    0    0   0    0   0    0    0       0    0   0    0   0    0   0   0   
1    0    0   0    0   0    0    0       0    0   0    0   0    0   0   0   
2    0    0   0    0   0    0    0       0    0   0    0   0    0   0   0   
3    0    0   0    0   0    0    0       0    0   0    1   0    0   0   0   
4    0    0   0    0   0    0    0       0    0   0    0   0    0   0   0   

   付连智  任达华  任静  任鹏远  伊一  伊春德  伊能静  伊莎贝尔·于佩尔  优恵  何念  何炅  何琳  余彬  余文乐  余男  \
0    0    0   0    0   0    0    0         0   0   0   0   0   0    0   0   
1    0    0   0    0   0    0    0         0   0   0   0   0   0    0   0   
2    0    0   0    0   0    0    0         0   0   0   0   0   0    0   0   
3    0    0   0    0   0    0    0         0   0   0   0   0   0    0   0   
4    0    0   0    0   0    0    0         0   1   0   0   0   0    0   0   

   佟丽娅  佟大为  佟瑞欣  侯勇  侯梦莎  保剑锋  俞杭英  倪虹洁  傅东育  傅彪  傅浤鸣  傅艺伟  克里斯·帕拉特  ...  \
0    0    0    0   0    0    0    0    0    0   0    0    0        0  ...   
1    0    0    0   0    0    0    0    0    0   0    0    0        0  ...   
2    0    0    0   0    0    0    0    0    0   0    0    0        0  ...   
3    0    0    0   0    0    0    0    0    0   0    0    0        0  ...   
4    0    0    0   0    0    0    0    0    0   0    0    0        0  ...   

   陈正道  陈红  陈继铭  陈逸宁  陶慧  陶晶莹  陶白莉  陶虹  隋兰  雷佳音  雷恪生  雷蒙德·雷德  霍建起  鞠觉亮  韩三平  \
0    0   0    0    0   0    0    0   0   0    0    0       0    0    0    0   
1    0   0    0    0   0    0    0   0   0    0    0       0    0    0    0   
2    0   0    0    0   0    0    0   0   0    0    0       0    0    0    0   
3    0   0    0    0   0    0    0   0   0    1    0       0    0    0    0   
4    0   0    0    0   0    0    0   0   0    0    0       0    0    0    0   

   韩东君  韩庚  韩昊霖  颜丙燕  马东  马修·莫里森  马健  马少骅  马思纯  马晓伟  马特·弗里沃  马苏  高一功  高圆圆  \
0    0   0    0    0   0       0   0    0    0    0       0   0    0    0   
1    0   0    1    0   0       0   0    0    0    0       0   0    0    0   
2    0   0    0    0   0       0   0    0    0    0       0   0    0    0   
3    0   0    0    0   0       1   0    0    0    0       0   0    0    0   
4    0   0    0    0   0       0   0    0    0    0       0   0    0    0   

   高宝宝  高捷  魏宗万  魏积安  鲍国安  麦斯·米科尔森  黄奕  黄宏  黄小蕾  黄尧  黄建新  黄晓明  黄梅莹  黄渤  黄磊  \
0    0   0    0    0    0        0   0   0    0   0    0    0    1   0   0   
1    0   0    0    0    0        0   0   0    0   0    0    0    0   1   0   
2    0   0    0    0    0        0   0   0    0   0    0    0    0   0   0   
3    0   0    0    0    0        0   0   0    0   0    0    0    0   1   0   
4    0   0    0    0    0        0   0   0    0   0    0    0    0   1   0   

   黄蜀芹  黄轩  黄达亮  黄龄  黎明  黑泽清  
0    0   0    0   0   0    0  
1    0   0    0   0   0    0  
2    0   0    0   0   0    0  
3    0   0    0   0   0    0  
4    0   0    0   0   0    0  

[5 rows x 483 columns]
-------------------- 频繁项集 --------------------
     support      itemsets
2   0.768421          (徐峥)
7   0.157895          (黄渤)
11  0.094737      (徐峥, 黄渤)
1   0.073684          (宁浩)
0   0.063158         (于和伟)
6   0.063158          (陶虹)
8   0.063158      (宁浩, 徐峥)
9   0.063158      (宁浩, 黄渤)
13  0.063158  (宁浩, 徐峥, 黄渤)
3   0.052632         (王宝强)
4   0.052632          (王迅)
5   0.052632         (陈凯歌)
10  0.052632      (徐峥, 陶虹)
12  0.052632      (黄渤, 王迅)
-------------------- 关联规则 --------------------
  antecedents consequents  antecedent support  consequent support   support  \
3    (徐峥, 黄渤)        (宁浩)            0.094737            0.073684  0.063158   
4        (宁浩)    (徐峥, 黄渤)            0.073684            0.094737  0.063158   
2    (宁浩, 徐峥)        (黄渤)            0.063158            0.157895  0.063158   
7        (王迅)        (黄渤)            0.052632            0.157895  0.052632   
5        (黄渤)    (宁浩, 徐峥)            0.157895            0.063158  0.063158   
6        (黄渤)        (王迅)            0.157895            0.052632  0.052632   
0        (宁浩)        (黄渤)            0.073684            0.157895  0.063158   
1        (黄渤)        (宁浩)            0.157895            0.073684  0.063158   

   confidence      lift  leverage  conviction  
3    0.666667  9.047619  0.056177    2.778947  
4    0.857143  9.047619  0.056177    6.336842  
2    1.000000  6.333333  0.053186         inf  
7    1.000000  6.333333  0.044321         inf  
5    0.400000  6.333333  0.053186    1.561404  
6    0.333333  6.333333  0.044321    1.421053  
0    0.857143  5.428571  0.051524    5.894737  
1    0.400000  5.428571  0.051524    1.543860  