import abnomal 
'''
result1 = abnomal.ab_by_date("2021-03-15")
print("整体检测结果（当日秒序号）:", result1)
'''
results = abnomal.ab_all_dates(method=["kmeans"])

results = abnomal.ab_all_dates(method=["knn"])

#results = abnomal.ab_all_dates(method=["isolateForest"])