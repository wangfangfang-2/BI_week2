from efficient_apriori import apriori
# 设置数据集
transactions = [('牛奶','面包','尿布'),
		('可乐','面包', '尿布', '啤酒'),
		('牛奶','尿布', '啤酒', '鸡蛋'),
		('面包', '牛奶', '尿布', '啤酒'),
		('面包', '牛奶', '尿布', '可乐')]
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)
print("频繁项集：", itemsets)
print("关联规则：", rules)

#以下为结果部分
频繁项集： {1: {('牛奶',): 4, ('面包',): 4, ('尿布',): 5, ('啤酒',): 3}, 2: {('尿布', '牛奶'): 4, ('尿布', '面包'): 4, ('牛奶', '面包'): 3, ('啤酒', '尿布'): 3}, 3: {('尿布', '牛奶', '面包'): 3}}
关联规则： [{牛奶} -> {尿布}, {面包} -> {尿布}, {啤酒} -> {尿布}, {牛奶, 面包} -> {尿布}