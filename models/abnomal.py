import pandas as pd
from sklearn.ensemble import IsolationForest
import os
from datetime import datetime, timedelta

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def getRes(resList,startNum):
    res1 = []
    start = 0
    for i in range(len(resList)-1):    
        if resList[i+1]-resList[i]>10:
            # 确保返回的是Python int类型
            res1.append([int(resList[start]+startNum), int(resList[i]+startNum)])
            start = i+1
    return res1

def get_day_info_from_data(df, date_str):
    """获取指定日期在数据中的起始索引、结束索引和所有索引列表"""
    try:
        if '记录仪时间' not in df.columns:
            return None, None, []
        
        df_copy = df.copy()
        df_copy['date_only'] = pd.to_datetime(df_copy['记录仪时间']).dt.date
        target_date = pd.to_datetime(date_str).date()
        day_mask = df_copy['date_only'] == target_date
        
        if not day_mask.any():
            return None, None, []
        
        day_indices = df_copy[day_mask].index.tolist()
        start_idx = min(day_indices)
        end_idx = max(day_indices)
        
        return start_idx, end_idx, day_indices
        
    except Exception as e:
        logger.error(f"获取日期 {date_str} 信息失败: {e}")
        return None, None, []

def convert_index_to_date_and_seconds_from_data(index, df):
    """从数据表中查询索引对应的日期和每日对应秒序号"""
    try:
        # 确保索引在有效范围内
        if index < 0 or index >= len(df):
            return None, None
        
        # 从数据表中获取对应行的数据
        row = df.iloc[index]
        
        # 获取日期
        if '记录仪时间' in df.columns:
            date_str = pd.to_datetime(row['记录仪时间']).strftime('%Y-%m-%d')
        elif 'date' in df.columns:
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        else:
            return None, None
        
        # 获取每日对应秒序号
        if '每日对应秒序号' in df.columns:
            seconds_of_day = int(row['每日对应秒序号'])
        elif 'time' in df.columns:
            seconds_of_day = int(row['time'])
        else:
            # 如果没有秒序号列，通过索引计算
            day_start_idx, _, _ = get_day_info_from_data(df, date_str)
            if day_start_idx is not None:
                seconds_of_day = index - day_start_idx
            else:
                return None, None
        
        return date_str, seconds_of_day
        
    except Exception as e:
        logger.error(f"从数据表查询索引 {index} 失败: {e}")
        return None, None

def convert_date_seconds_to_index_from_data(date_str, start_seconds=None, end_seconds=None, df=None):
    """从数据表中查询日期和每日对应秒序号对应的索引"""
    if df is None:
        return None, None
    
    try:
        day_start_idx, day_end_idx, day_indices = get_day_info_from_data(df, date_str)
        if day_start_idx is None:
            return None, None
        
        # 如果有每日对应秒序号列，直接查找
        time_col = None
        if '每日对应秒序号' in df.columns:
            time_col = '每日对应秒序号'
        elif 'time' in df.columns:
            time_col = 'time'
        
        if time_col:
            day_data = df.loc[day_indices, time_col]
            
            if start_seconds is None:
                start_index = day_indices[0]
            else:
                # 查找最接近start_seconds的索引
                closest_start = day_data.iloc[(day_data - start_seconds).abs().argsort()[:1]]
                start_index = closest_start.index[0] if len(closest_start) > 0 else day_indices[0]
            
            if end_seconds is None:
                end_index = day_indices[-1]
            else:
                # 查找最接近end_seconds的索引
                closest_end = day_data.iloc[(day_data - end_seconds).abs().argsort()[:1]]
                end_index = closest_end.index[0] if len(closest_end) > 0 else day_indices[-1]
        else:
            # 如果没有秒序号列，使用相对位置
            if start_seconds is None:
                start_index = day_indices[0]
            else:
                start_index = min(day_start_idx + start_seconds, day_indices[-1])
            
            if end_seconds is None:
                end_index = day_indices[-1]
            else:
                end_index = min(day_start_idx + end_seconds, day_indices[-1])
        
        return start_index, end_index
        
    except Exception as e:
        logger.error(f"转换日期秒序号到索引失败: {e}")
        return None, None

def convert_index_to_date_and_seconds(index, base_date="2021-03-14"):
    """将索引转换为日期和每日对应秒序号（备用方法）"""
    base_datetime = datetime.strptime(base_date, "%Y-%m-%d")
    target_datetime = base_datetime + timedelta(seconds=int(index))
    date_str = target_datetime.strftime("%Y-%m-%d")
    
    # 计算每日对应秒序号（从当天0点开始的秒数）
    start_of_day = datetime.strptime(date_str, "%Y-%m-%d")
    seconds_of_day = int((target_datetime - start_of_day).total_seconds())
    
    return date_str, seconds_of_day

def remove_overlapping_intervals(intervals):
    """去除重叠的时间区间"""
    if not intervals:
        return []
    
    # 按起始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        
        # 如果当前区间与上一个区间重叠或相邻
        if current[0] <= last[1] + 1:
            # 合并区间
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # 不重叠，添加新区间
            merged.append(current)
    
    return merged

def save_results_to_csv(results, output_file="iForest.csv", df=None):
    """将结果保存到CSV文件（为每个方法创建单独的文件）"""
    saved_files = []
    
    for method, intervals in results.items():
        # 为每个方法生成不同的文件名
        method_file = f"{method}.csv"
        csv_data = []
        
        for start_idx, end_idx in intervals:
            if df is not None:
                # 从数据表中查询
                start_date, start_seconds = convert_index_to_date_and_seconds_from_data(start_idx, df)
                end_date, end_seconds = convert_index_to_date_and_seconds_from_data(end_idx, df)
            else:
                # 使用备用计算方法
                start_date, start_seconds = convert_index_to_date_and_seconds(start_idx)
                end_date, end_seconds = convert_index_to_date_and_seconds(end_idx)
            
            if start_date is None or end_date is None:
                continue
            
            # 如果跨天，需要分别处理
            if start_date == end_date:
                csv_data.append({
                    "日期": start_date,
                    "每日对应秒序号起始": start_seconds,
                    "每日对应秒序号结束": end_seconds,
                    "异常类型": ''
                })
            else:
                # 跨天情况处理
                if df is not None:
                    # 获取第一天的结束秒序号和第二天的开始秒序号
                    _, day_end_idx, _ = get_day_info_from_data(df, start_date)
                    day_start_idx, _, _ = get_day_info_from_data(df, end_date)
                    
                    if day_end_idx is not None:
                        _, first_day_end_seconds = convert_index_to_date_and_seconds_from_data(day_end_idx, df)
                    else:
                        first_day_end_seconds = 86399
                    
                    if day_start_idx is not None:
                        _, second_day_start_seconds = convert_index_to_date_and_seconds_from_data(day_start_idx, df)
                    else:
                        second_day_start_seconds = 0
                else:
                    first_day_end_seconds = 86399
                    second_day_start_seconds = 0
                
                # 第一天记录
                csv_data.append({
                    "日期": start_date,
                    "每日对应秒序号起始": start_seconds,
                    "每日对应秒序号结束": first_day_end_seconds,
                    "异常类型": ''
                })
                # 第二天记录
                csv_data.append({
                    "日期": end_date,
                    "每日对应秒序号起始": second_day_start_seconds,
                    "每日对应秒序号结束": end_seconds,
                    "异常类型": ''
                })
        
        if not csv_data:
            logger.info(f"方法 {method} 没有检测到异常，跳过文件保存")
            continue
        
        # 创建新数据的DataFrame
        new_df = pd.DataFrame(csv_data)
        output_path = os.path.join(os.path.dirname(__file__), "..", "timeEvalWebData", method_file)
        
        # 检查文件是否存在
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path, encoding='utf-8-sig')
                logger.info(f"读取现有文件 {method_file}，包含 {len(existing_df)} 条记录")
                
                # 合并新旧数据
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # 强化去重逻辑
                before_dedup = len(combined_df)
                
                # 确保数据类型一致，处理可能的空值和数据类型问题
                combined_df['日期'] = combined_df['日期'].astype(str).str.strip()
                combined_df['每日对应秒序号起始'] = pd.to_numeric(combined_df['每日对应秒序号起始'], errors='coerce').fillna(0).astype(int)
                combined_df['每日对应秒序号结束'] = pd.to_numeric(combined_df['每日对应秒序号结束'], errors='coerce').fillna(0).astype(int)
                #combined_df['异常类型'] = combined_df['异常类型'].fillna(method).astype(str).str.strip()
                
                # 基于关键列去重，使用更严格的去重策略
                combined_df = combined_df.drop_duplicates(
                    subset=['日期', '每日对应秒序号起始', '每日对应秒序号结束'],
                    keep='first'
                ).reset_index(drop=True)
                
                after_dedup = len(combined_df)
                
                if before_dedup > after_dedup:
                    logger.info(f"方法 {method}: 去除了 {before_dedup - after_dedup} 条重复记录")
                
                # 按日期和起始时间排序
                combined_df = combined_df.sort_values(['日期', '每日对应秒序号起始'])
                
                # 保存合并后的数据
                combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"方法 {method}: 增量保存完成，总共 {len(combined_df)} 条记录（新增 {len(new_df)} 条）")
                
            except Exception as e:
                logger.error(f"读取现有文件 {method_file} 失败: {e}，将创建新文件")
                new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"方法 {method}: 创建新文件，保存 {len(new_df)} 条记录")
        else:
            new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"方法 {method}: 创建新文件，保存 {len(new_df)} 条记录")
        
        saved_files.append(output_path)
    
    return saved_files

def ab_by_date(date_str, start_seconds=None, end_seconds=None, method=["isolateForest"], save_to_csv=True):
    """基于日期和每日对应秒序号进行异常检测"""
    # 加载数据
    file_path = os.path.join(os.path.dirname(__file__), "..", "timeEvalWebData", "month_3_processed.csv")
    df = pd.read_csv(file_path)
    
    # 使用转换函数
    start_index, end_index = convert_date_seconds_to_index_from_data(date_str, start_seconds, end_seconds, df)
    
    if start_index is None or end_index is None:
        logger.error(f"无法找到日期 {date_str} 对应的数据")
        return {}
    
    logger.info(f"检测日期: {date_str}, 数据索引范围: {start_index} - {end_index}")
    
    # 调用异常检测函数
    result = ab(start_index, end_index, method, save_to_csv=False)
    
    # 将结果转换为当日对应秒序号
    converted_result = {}
    for method_name, intervals in result.items():
        converted_intervals = []
        for start_idx, end_idx in intervals:
            # 从数据表中查询秒序号
            start_date, start_seconds_of_day = convert_index_to_date_and_seconds_from_data(start_idx, df)
            end_date, end_seconds_of_day = convert_index_to_date_and_seconds_from_data(end_idx, df)
            
            if start_date is None or end_date is None:
                continue
            
            # 只保留指定日期的结果
            if start_date == date_str:
                if end_date == date_str:
                    converted_intervals.append([start_seconds_of_day, end_seconds_of_day])
                else:
                    # 跨天的异常，查询当天的最后秒序号
                    _, day_end_idx, _ = get_day_info_from_data(df, start_date)
                    if day_end_idx is not None:
                        _, last_seconds = convert_index_to_date_and_seconds_from_data(day_end_idx, df)
                        converted_intervals.append([start_seconds_of_day, last_seconds])
        
        if converted_intervals:
            converted_intervals = remove_overlapping_intervals(converted_intervals)
            converted_result[method_name] = converted_intervals
    
    # 如果需要保存到CSV
    if save_to_csv and result:
        save_results_to_csv(result, df=df)
    
    return converted_result

def ab(start:int, end:int, method:list, save_to_csv=True):
    """异常检测函数（基于索引）- 支持多种检测方法"""
    file_path = os.path.join(os.path.dirname(__file__), "..", "timeEvalWebData", "month_3_processed.csv")
    
    df = pd.read_csv(file_path)
    df = df[["记录仪时间","8井油压","9井压力","9井流量"]]
    df.columns = ["date","p8","p9","f9"]
    data = df[start: end]
    # 修复fillna的deprecated警告
    data = data.ffill(axis=0)
    data = data.reset_index()
    data = data[["p8","p9","f9"]]
    logger.info("Finish Data load")
    
    res = {}
    
    if "isolateForest" in method:
        from sklearn.ensemble import IsolationForest
        isomodel= IsolationForest(n_estimators=100, 
                      max_samples='auto', 
                      contamination=float(0.01),
                      random_state=42
                     )
        isomodel.fit(data)
        score = isomodel.fit_predict(data)
        res_iso = [i for i in range(len(score)) if score[i] == -1]
        res["isolateForest"] = getRes(res_iso, start)
        logger.info("Finish Isolate Forest")
    
    if "kmeans" in method:
        from kmeans import KMeansAD
        detector = KMeansAD(k=20,n_jobs=1,stride=1,window_size=20)
        scores = detector.fit_predict(data)
        resKmeans = [i for i in range(len(scores)) if scores[i]>10]
        res["kmeans"] = getRes(resKmeans, start)
        logger.info("Finish Kmeans")
    
    if "knn" in method:
        from pyod.models.knn import KNN
        clf = KNN(
        # n_neighbors=5,
        # method="largest",
        # radius=1.0,
        # leaf_size=30,
        # n_jobs=1,
        # algorithm="auto",
        # metric=2,
        # metric_params=None,
        # p=2,
        )
        clf.fit(data)
        scores = clf.decision_scores_
        resKNN = [i for i in range(len(scores)) if scores[i]>0]
        res["knn"] = getRes(resKNN, start)
        logger.info("Finish knn")
    
    if "lstm" in method:
        from lstm_ad.model import LSTMAD
        # 修改模型路径为相对路径
        model_path = os.path.join(os.path.dirname(__file__), "lstm_res", "1344.pt")
        model = LSTMAD.load(path = model_path,
                            input_size=data.shape[1],
                            lstm_layers=2,
                             split=0.9,
                             window_size=30,
                             prediction_window_size=1,
                             output_dims=[],
                             batch_size=32,
                             validation_batch_size=128,
                             test_batch_size=128,
                             epochs=50,
                             early_stopping_delta=0.05,
                             early_stopping_patience=10,
                             optimizer='adam',
                             learning_rate=0.001,
                             random_state=42)
        scores = model.anomaly_detection(data)
        resLstm = [i for i in range(len(scores)) if scores[i]>3]
        res["lstm"] = getRes(resLstm, start)
        logger.info("Finish LSTM-ad")

    # 如果需要保存到CSV
    if save_to_csv and res:
        save_results_to_csv(res)

    return res


def get_all_dates_from_data():
    """获取数据中所有可用的日期列表"""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "..", "timeEvalWebData", "month_3_processed.csv")
        df = pd.read_csv(file_path)
        
        if '记录仪时间' not in df.columns:
            logger.error("数据文件中没有找到'记录仪时间'列")
            return []
        
        # 提取所有唯一日期
        df_copy = df.copy()
        df_copy['date_only'] = pd.to_datetime(df_copy['记录仪时间']).dt.date
        unique_dates = sorted(df_copy['date_only'].unique())
        
        # 转换为字符串格式
        date_strings = [date.strftime('%Y-%m-%d') for date in unique_dates]
        
        logger.info(f"找到 {len(date_strings)} 个可用日期: {date_strings[0]} 到 {date_strings[-1]}")
        return date_strings
        
    except Exception as e:
        logger.error(f"获取日期列表失败: {e}")
        return []

def ab_all_dates(method=["isolateForest"], save_to_csv=True, start_date=None, end_date=None):
    """
    遍历所有可用日期进行异常检测
    
    Args:
        method (list): 异常检测方法列表，默认为["isolateForest"]
        save_to_csv (bool): 是否保存结果到CSV文件，默认为True
        start_date (str): 开始日期，格式为'YYYY-MM-DD'，默认为None（从最早日期开始）
        end_date (str): 结束日期，格式为'YYYY-MM-DD'，默认为None（到最晚日期结束）
    
    Returns:
        dict: 所有日期的异常检测结果，格式为 {date: {method: intervals}}
    """
    logger.info("开始遍历所有日期进行异常检测...")
    
    # 获取所有可用日期
    all_dates = get_all_dates_from_data()
    
    if not all_dates:
        logger.error("没有找到可用的日期数据")
        return {}
    
    # 过滤日期范围
    filtered_dates = all_dates
    if start_date:
        filtered_dates = [date for date in filtered_dates if date >= start_date]
    if end_date:
        filtered_dates = [date for date in filtered_dates if date <= end_date]
    
    if not filtered_dates:
        logger.error(f"指定的日期范围内没有可用数据: {start_date} 到 {end_date}")
        return {}
    
    logger.info(f"将处理 {len(filtered_dates)} 个日期: {filtered_dates[0]} 到 {filtered_dates[-1]}")
    
    all_results = {}
    success_count = 0
    error_count = 0
    
    for i, date_str in enumerate(filtered_dates, 1):
        try:
            logger.info(f"正在处理第 {i}/{len(filtered_dates)} 个日期: {date_str}")
            
            # 调用ab_by_date进行异常检测
            result = ab_by_date(date_str, method=method, save_to_csv=save_to_csv)
            
            if result:
                all_results[date_str] = result
                success_count += 1
                
                # 输出当前日期的检测结果摘要
                for method_name, intervals in result.items():
                    logger.info(f"  {date_str} - {method_name}: 检测到 {len(intervals)} 个异常区间")
            else:
                logger.warning(f"  {date_str}: 未检测到异常或处理失败")
                error_count += 1
                
        except Exception as e:
            logger.error(f"处理日期 {date_str} 时出错: {e}")
            error_count += 1
            continue
    
    # 输出总结信息
    logger.info(f"异常检测完成！")
    logger.info(f"成功处理: {success_count} 个日期")
    logger.info(f"处理失败: {error_count} 个日期")
    logger.info(f"总异常日期数: {len(all_results)} 个")
    
    # 统计总的异常区间数
    total_intervals = 0
    for date_results in all_results.values():
        for method_intervals in date_results.values():
            total_intervals += len(method_intervals)
    
    logger.info(f"总异常区间数: {total_intervals} 个")
    
    if save_to_csv:
        output_path = os.path.join(os.path.dirname(__file__), "..", "timeEvalWebData", "iForest.csv")
        logger.info(f"所有结果已保存到: {output_path}")
    
    return all_results

def ab_date_range(start_date, end_date, method=["isolateForest"], save_to_csv=True):
    """
    对指定日期范围进行异常检测（ab_all_dates的简化版本）
    
    Args:
        start_date (str): 开始日期，格式为'YYYY-MM-DD'
        end_date (str): 结束日期，格式为'YYYY-MM-DD'
        method (list): 异常检测方法列表，默认为["isolateForest"]
        save_to_csv (bool): 是否保存结果到CSV文件，默认为True
    
    Returns:
        dict: 指定日期范围的异常检测结果
    """
    return ab_all_dates(method=method, save_to_csv=save_to_csv, start_date=start_date, end_date=end_date)