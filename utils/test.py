import time_filter
# 创建筛选器实例
filter_tool = time_filter.TimeRangeFilter()
    
# 示例1: 筛选3月15号10:00到12:00的数据
print("=== 示例1: 筛选3月15号10:00到12:00的数据 ===")
result1 = filter_tool.filter_single_day(
    date='2021-03-15',
    start_time='10:00',
    end_time='12:00'
)
print(f"筛选结果: {len(result1)} 条记录")
if len(result1) > 0:
    print("前5条记录:")
    print(result1.head())
    
# 示例2: 筛选跨日期的时间范围
print("\n=== 示例2: 筛选3月15号23:00到3月16号01:00的数据 ===")
result2 = filter_tool.filter_by_time_range(
    start_date='2021-03-15',
    start_time='23:00',
    end_date='2021-03-16',
    end_time='01:00'
)
print(f"筛选结果: {len(result2)} 条记录")
    
# 示例3: 只输出特定列
print("\n=== 示例3: 只输出特定列 ===")
result3 = filter_tool.filter_single_day(
    date='2021-03-15',
    start_time='10:00',
    end_time='12:00',
    output_columns=['日期', '小时', '分钟', '8井油压', '9井压力']
)
print(f"筛选结果: {len(result3)} 条记录")
if len(result3) > 0:
    print("前5条记录:")
    print(result3.head())
    
# 示例4: 查看可用日期
print("\n=== 示例4: 查看可用日期 ===")
available_dates = filter_tool.get_available_dates()
print(f"可用日期: {available_dates[:10]}...")  # 只显示前10个
    
# 示例5: 查看指定日期的时间范围
print("\n=== 示例5: 查看指定日期的时间范围 ===")
start_time, end_time = filter_tool.get_time_range_for_date('2021-03-14')
print(f"2021-03-15的时间范围: {start_time} - {end_time}")