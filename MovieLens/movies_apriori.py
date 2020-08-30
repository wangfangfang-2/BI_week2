# 分析MovieLens 电影分类中的频繁项集和关联规则
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 数据加载
movies = pd.read_csv('./movies.csv')
#print(movies.head())
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
print(movies['genres'])
movies_hot_encoded = movies.drop('genres',1).join(movies.genres.str.get_dummies(sep='|'))
print(movies_hot_encoded)

pd.options.display.max_columns=100
print(movies_hot_encoded.head())

# 将movieId, title设置为index
movies_hot_encoded.set_index(['movieId','title'],inplace=True)
#print(movies_hot_encoded.head())
# 挖掘频繁项集，最小支持度为0.02
itemsets = apriori(movies_hot_encoded,use_colnames=True, min_support=0.02)
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
0        Adventure|Animation|Children|Comedy|Fantasy
1                         Adventure|Children|Fantasy
2                                     Comedy|Romance
3                               Comedy|Drama|Romance
4                                             Comedy
                            ...                     
27273                                         Comedy
27274                                         Comedy
27275                                      Adventure
27276                             (no genres listed)
27277                       Adventure|Fantasy|Horror
Name: genres, Length: 27278, dtype: object
       movieId                               title  (no genres listed)  Action  Adventure  Animation  Children  Comedy  Crime  Documentary  Drama  Fantasy  Film-Noir  Horror  IMAX  Musical  Mystery  Romance  Sci-Fi  Thriller  War  Western
0            1                    Toy Story (1995)                   0       0          1          1         1       1      0            0      0        1          0       0     0        0        0        0       0         0    0        0
1            2                      Jumanji (1995)                   0       0          1          0         1       0      0            0      0        1          0       0     0        0        0        0       0         0    0        0
2            3             Grumpier Old Men (1995)                   0       0          0          0         0       1      0            0      0        0          0       0     0        0        0        1       0         0    0        0
3            4            Waiting to Exhale (1995)                   0       0          0          0         0       1      0            0      1        0          0       0     0        0        0        1       0         0    0        0
4            5  Father of the Bride Part II (1995)                   0       0          0          0         0       1      0            0      0        0          0       0     0        0        0        0       0         0    0        0
...        ...                                 ...                 ...     ...        ...        ...       ...     ...    ...          ...    ...      ...        ...     ...   ...      ...      ...      ...     ...       ...  ...      ...
27273   131254        Kein Bund für's Leben (2007)                   0       0          0          0         0       1      0            0      0        0          0       0     0        0        0        0       0         0    0        0
27274   131256       Feuer, Eis & Dosenbier (2002)                   0       0          0          0         0       1      0            0      0        0          0       0     0        0        0        0       0         0    0        0
27275   131258                  The Pirates (2014)                   0       0          1          0         0       0      0            0      0        0          0       0     0        0        0        0       0         0    0        0
27276   131260                 Rentun Ruusu (2001)                   1       0          0          0         0       0      0            0      0        0          0       0     0        0        0        0       0         0    0        0
27277   131262                    Innocence (2014)                   0       0          1          0         0       0      0            0      0        1          0       1     0        0        0        0       0         0    0        0

[27278 rows x 22 columns]
   movieId                               title  (no genres listed)  Action  \
0        1                    Toy Story (1995)                   0       0   
1        2                      Jumanji (1995)                   0       0   
2        3             Grumpier Old Men (1995)                   0       0   
3        4            Waiting to Exhale (1995)                   0       0   
4        5  Father of the Bride Part II (1995)                   0       0   

   Adventure  Animation  Children  Comedy  Crime  Documentary  Drama  Fantasy  \
0          1          1         1       1      0            0      0        1   
1          1          0         1       0      0            0      0        1   
2          0          0         0       1      0            0      0        0   
3          0          0         0       1      0            0      1        0   
4          0          0         0       1      0            0      0        0   

   Film-Noir  Horror  IMAX  Musical  Mystery  Romance  Sci-Fi  Thriller  War  \
0          0       0     0        0        0        0       0         0    0   
1          0       0     0        0        0        0       0         0    0   
2          0       0     0        0        0        1       0         0    0   
3          0       0     0        0        0        1       0         0    0   
4          0       0     0        0        0        0       0         0    0   

   Western  
0        0  
1        0  
2        0  
3        0  
4        0  
-------------------- 频繁项集 --------------------
     support                  itemsets
7   0.489185                   (Drama)
4   0.306987                  (Comedy)
14  0.153164                (Thriller)
12  0.151294                 (Romance)
0   0.129042                  (Action)
5   0.107743                   (Crime)
9   0.095718                  (Horror)
31  0.094325          (Drama, Romance)
26  0.093335           (Drama, Comedy)
6   0.090586             (Documentary)
1   0.085380               (Adventure)
27  0.069470         (Romance, Comedy)
32  0.068480         (Drama, Thriller)
13  0.063898                  (Sci-Fi)
28  0.062761            (Drama, Crime)
11  0.055503                 (Mystery)
8   0.051763                 (Fantasy)
29  0.045165         (Crime, Thriller)
20  0.044101           (Drama, Action)
15  0.043772                     (War)
3   0.041755                (Children)
22  0.040655        (Thriller, Action)
34  0.039336        (Thriller, Horror)
10  0.037979                 (Musical)
2   0.037649               (Animation)
17  0.035633       (Adventure, Action)
33  0.032774              (Drama, War)
35  0.029144       (Thriller, Mystery)
19  0.028118           (Crime, Action)
36  0.027458  (Drama, Comedy, Romance)
30  0.026432          (Drama, Mystery)
18  0.026358          (Comedy, Action)
25  0.025368           (Comedy, Crime)
24  0.025295        (Drama, Adventure)
37  0.024965  (Drama, Crime, Thriller)
16  0.024782                 (Western)
21  0.023499          (Sci-Fi, Action)
23  0.022032       (Adventure, Comedy)
-------------------- 关联规则 --------------------
          antecedents        consequents  antecedent support  \
9           (Mystery)         (Thriller)            0.055503   
8          (Thriller)          (Mystery)            0.153164   
14            (Crime)  (Drama, Thriller)            0.107743   
13  (Drama, Thriller)            (Crime)            0.068480   
7            (Action)        (Adventure)            0.129042   
6         (Adventure)           (Action)            0.085380   
16           (Sci-Fi)           (Action)            0.063898   
17           (Action)           (Sci-Fi)            0.129042   
1          (Thriller)            (Crime)            0.153164   
0             (Crime)         (Thriller)            0.107743   
5            (Horror)         (Thriller)            0.095718   
4          (Thriller)           (Horror)            0.153164   
12     (Drama, Crime)         (Thriller)            0.062761   
15         (Thriller)     (Drama, Crime)            0.153164   
3            (Action)         (Thriller)            0.129042   
2          (Thriller)           (Action)            0.153164   
10            (Crime)           (Action)            0.107743   
11           (Action)            (Crime)            0.129042   

    consequent support   support  confidence      lift  leverage  conviction  
9             0.153164  0.029144    0.525099  3.428352  0.020643    1.783185  
8             0.055503  0.029144    0.190282  3.428352  0.020643    1.166453  
14            0.068480  0.024965    0.231711  3.383632  0.017587    1.212461  
13            0.107743  0.024965    0.364561  3.383632  0.017587    1.404159  
7             0.085380  0.035633    0.276136  3.234198  0.024616    1.263525  
6             0.129042  0.035633    0.417347  3.234198  0.024616    1.494813  
16            0.129042  0.023499    0.367757  2.849906  0.015253    1.377568  
17            0.063898  0.023499    0.182102  2.849906  0.015253    1.144523  
1             0.107743  0.045165    0.294878  2.736877  0.028662    1.265394  
0             0.153164  0.045165    0.419190  2.736877  0.028662    1.458027  
5             0.153164  0.039336    0.410954  2.683100  0.024675    1.437639  
4             0.095718  0.039336    0.256821  2.683100  0.024675    1.216776  
12            0.153164  0.024965    0.397780  2.597093  0.015352    1.406192  
15            0.062761  0.024965    0.162997  2.597093  0.015352    1.119755  
3             0.153164  0.040655    0.315057  2.056994  0.020891    1.236360  
2             0.129042  0.040655    0.265438  2.056994  0.020891    1.185684  
10            0.129042  0.028118    0.260973  2.022393  0.014215    1.178520  
11            0.107743  0.028118    0.217898  2.022393  0.014215    1.140845 
