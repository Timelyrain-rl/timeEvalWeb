# -*- coding: utf-8 -*-
"""
主要功能：
- 数据预加载到内存以提高查询性能
- 支持按日期和时间范围筛选数据
- 实时数据可视化
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import json
import atexit
import models.abnomal as abnomal

# 创建Flask应用实例
app = Flask(__name__)

# 数据文件路径配置
DATA_FILE = os.path.join('timeEvalWebData', 'month_3_processed.csv')

# 全局变量：存储预加载的数据和日期范围
GLOBAL_DATA = None  # 存储完整的数据集
DATE_RANGE = None   # 存储可用的日期范围列表

# 在文件开头添加新的全局变量
ANNO_DATA = None    # 存储异常标注数据
IFOREST_DATA = None # 存储iForest异常检测数据
KMEAN_DATA = None    # 存储K-Means异常检测数据
LSTM_AD_DATA = None  # 存储LSTM异常检测数据
KNN_DATA = None      # 存储KNN异常检测数据

# 修改load_data_to_memory函数
def load_data_to_memory():
    """
    应用启动时预加载数据到内存
    
    将CSV文件中的数据加载到内存中，并进行必要的数据类型转换和预处理。
    这样可以避免每次查询时都读取文件，大大提高响应速度。
    
    Returns:
        bool: 数据加载是否成功
    """
    global GLOBAL_DATA, DATE_RANGE, ANNO_DATA, IFOREST_DATA, KMEAN_DATA, LSTM_AD_DATA, KNN_DATA
    try:
        print("正在加载数据到内存...")
        # 读取主要数据CSV文件
        GLOBAL_DATA = pd.read_csv(DATA_FILE)
        
        # 读取标注数据
        anno_file = os.path.join('timeEvalWebData', 'anno_data.csv')
        iforest_file = os.path.join('timeEvalWebData', 'iForest.csv')
        
        if os.path.exists(anno_file):
            ANNO_DATA = pd.read_csv(anno_file)
            # 处理日期格式
            ANNO_DATA['日期'] = pd.to_datetime(ANNO_DATA['日期']).dt.strftime('%Y-%m-%d')
            print(f"标注数据加载完成，共 {len(ANNO_DATA)} 条记录")
        else:
            print("标注数据文件不存在")
            ANNO_DATA = pd.DataFrame()
            
        if os.path.exists(iforest_file):
            IFOREST_DATA = pd.read_csv(iforest_file)
            # 处理日期格式
            IFOREST_DATA['日期'] = pd.to_datetime(IFOREST_DATA['日期']).dt.strftime('%Y-%m-%d')
            print(f"iForest数据加载完成，共 {len(IFOREST_DATA)} 条记录")
        else:
            print("iForest数据文件不存在")
            IFOREST_DATA = pd.DataFrame()
        
        # 读取K-Means数据
        kmean_file = os.path.join('timeEvalWebData', 'kmeans.csv')
        if os.path.exists(kmean_file):
            KMEAN_DATA = pd.read_csv(kmean_file)
            KMEAN_DATA['日期'] = pd.to_datetime(KMEAN_DATA['日期']).dt.strftime('%Y-%m-%d')
            print(f"K-Means数据加载成功: {len(KMEAN_DATA)} 条记录")
        else:
            KMEAN_DATA = pd.DataFrame()
            print("K-Means数据文件不存在")
        
        # 读取LSTM-AD数据
        lstm_ad_file = os.path.join('timeEvalWebData', 'lstm_ad.csv')
        if os.path.exists(lstm_ad_file):
            LSTM_AD_DATA = pd.read_csv(lstm_ad_file)
            LSTM_AD_DATA['日期'] = pd.to_datetime(LSTM_AD_DATA['日期']).dt.strftime('%Y-%m-%d')
            print(f"LSTM-AD数据加载成功: {len(LSTM_AD_DATA)} 条记录")
        else:
            LSTM_AD_DATA = pd.DataFrame()
            print("LSTM-AD数据文件不存在")
        
        # 读取KNN数据
        knn_file = os.path.join('timeEvalWebData', 'knn.csv')
        if os.path.exists(knn_file):
            KNN_DATA = pd.read_csv(knn_file)
            KNN_DATA['日期'] = pd.to_datetime(KNN_DATA['日期']).dt.strftime('%Y-%m-%d')
            print(f"KNN数据加载成功: {len(KNN_DATA)} 条记录")
        else:
            KNN_DATA = pd.DataFrame()
            print("KNN数据文件不存在")
        
        # 处理日期列：转换为标准格式并提取唯一日期
        if '日期' in GLOBAL_DATA.columns:
            GLOBAL_DATA['日期'] = pd.to_datetime(GLOBAL_DATA['日期']).dt.strftime('%Y-%m-%d')
            DATE_RANGE = sorted(GLOBAL_DATA['日期'].unique().tolist())
        else:
            raise ValueError('数据文件缺少日期列')
            
        # 确保时间列为整数类型
        if '小时' in GLOBAL_DATA.columns:
            GLOBAL_DATA['小时'] = GLOBAL_DATA['小时'].astype(int)
        if '分钟' in GLOBAL_DATA.columns:
            GLOBAL_DATA['分钟'] = GLOBAL_DATA['分钟'].astype(int)
            
        print(f"数据加载完成，共 {len(GLOBAL_DATA)} 条记录")
        print(f"可用日期范围: {DATE_RANGE[:5]}...{DATE_RANGE[-5:]}")
        return True
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False

# 添加新的API接口获取标注数据
@app.route('/get_annotations')
def get_annotations():
    try:
        if ANNO_DATA is None or IFOREST_DATA is None or KMEAN_DATA is None or LSTM_AD_DATA is None or KNN_DATA is None:
            if not load_data_to_memory():
                return jsonify({'error': '数据加载失败，请检查数据文件'})
        
        date = request.args.get('date')
        if not date:
            return jsonify({'error': '缺少日期参数'})
        
        # 获取指定日期的标注数据
        anno_filtered = ANNO_DATA[ANNO_DATA['日期'] == date] if not ANNO_DATA.empty else pd.DataFrame()
        iforest_filtered = IFOREST_DATA[IFOREST_DATA['日期'] == date] if not IFOREST_DATA.empty else pd.DataFrame()
        kmean_filtered = KMEAN_DATA[KMEAN_DATA['日期'] == date] if not KMEAN_DATA.empty else pd.DataFrame()
        lstm_ad_filtered = LSTM_AD_DATA[LSTM_AD_DATA['日期'] == date] if not LSTM_AD_DATA.empty else pd.DataFrame()
        knn_filtered = KNN_DATA[KNN_DATA['日期'] == date] if not KNN_DATA.empty else pd.DataFrame()
        
        # 处理标注数据
        annotations = []
        
        # 处理anno_data中的标注
        for _, row in anno_filtered.iterrows():
            annotation = {
                'type': 'anno',
                'start': int(row['每日对应秒序号起始']) if pd.notna(row['每日对应秒序号起始']) else None,
                'end': int(row['每日对应秒序号结束']) if pd.notna(row['每日对应秒序号结束']) else None,
                'label': str(row['异常类型']) if pd.notna(row['异常类型']) else '',
                'date': row['日期']
            }
            annotations.append(annotation)
        
        # 处理iForest中的标注
        for _, row in iforest_filtered.iterrows():
            anomaly_type = str(row['异常类型']) if pd.notna(row['异常类型']) and str(row['异常类型']).strip() != '' else ''
            if anomaly_type:
                label = f'iForest-{anomaly_type}'
            else:
                label = 'iForest'
            
            annotation = {
                'type': 'iforest',
                'start': int(row['每日对应秒序号起始']) if pd.notna(row['每日对应秒序号起始']) else None,
                'end': int(row['每日对应秒序号结束']) if pd.notna(row['每日对应秒序号结束']) else None,
                'label': label,
                'date': row['日期']
            }
            annotations.append(annotation)
        
        # 处理K-Means中的标注
        for _, row in kmean_filtered.iterrows():
            anomaly_type = str(row['异常类型']) if pd.notna(row['异常类型']) and str(row['异常类型']).strip() != '' else ''
            if anomaly_type:
                label = f'K-Means-{anomaly_type}'
            else:
                label = 'K-Means'
            
            annotation = {
                'type': 'kmean',
                'start': int(row['每日对应秒序号起始']) if pd.notna(row['每日对应秒序号起始']) else None,
                'end': int(row['每日对应秒序号结束']) if pd.notna(row['每日对应秒序号结束']) else None,
                'label': label,
                'date': row['日期']
            }
            annotations.append(annotation)
        
        # 处理LSTM-AD中的标注
        for _, row in lstm_ad_filtered.iterrows():
            anomaly_type = str(row['异常类型']) if pd.notna(row['异常类型']) and str(row['异常类型']).strip() != '' else ''
            if anomaly_type:
                label = f'LSTM-{anomaly_type}'
            else:
                label = 'LSTM'
            
            annotation = {
                'type': 'lstm-ad',
                'start': int(row['每日对应秒序号起始']) if pd.notna(row['每日对应秒序号起始']) else None,
                'end': int(row['每日对应秒序号结束']) if pd.notna(row['每日对应秒序号结束']) else None,
                'label': label,
                'date': row['日期']
            }
            annotations.append(annotation)
        
        # 处理KNN中的标注
        for _, row in knn_filtered.iterrows():
            anomaly_type = str(row['异常类型']) if pd.notna(row['异常类型']) and str(row['异常类型']).strip() != '' else ''
            if anomaly_type:
                label = f'KNN-{anomaly_type}'
            else:
                label = 'KNN'
            
            annotation = {
                'type': 'knn',
                'start': int(row['每日对应秒序号起始']) if pd.notna(row['每日对应秒序号起始']) else None,
                'end': int(row['每日对应秒序号结束']) if pd.notna(row['每日对应秒序号结束']) else None,
                'label': label,
                'date': row['日期']
            }
            annotations.append(annotation)
        
        return jsonify({'annotations': annotations})
    except Exception as e:
        print(f"获取标注数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

# 修改cleanup函数
def cleanup():
    """
    应用关闭时的清理工作
    
    释放内存中的数据，确保应用优雅关闭。
    """
    global GLOBAL_DATA, DATE_RANGE, ANNO_DATA, IFOREST_DATA, KMEAN_DATA, LSTM_AD_DATA, KNN_DATA
    GLOBAL_DATA = None
    DATE_RANGE = None
    ANNO_DATA = None
    IFOREST_DATA = None
    KMEAN_DATA = None
    LSTM_AD_DATA = None
    KNN_DATA = None
    print("应用关闭，内存已清理")

def get_filtered_data(date, start_hour, end_hour, start_minute, end_minute):
    """
    根据指定条件筛选数据
    
    Args:
        date (str): 目标日期，格式为 'YYYY-MM-DD'
        start_hour (int): 开始小时 (0-23)
        end_hour (int): 结束小时 (0-23)
        start_minute (int): 开始分钟 (0-59)
        end_minute (int): 结束分钟 (0-59)
    
    Returns:
        dict: 包含筛选后数据的字典，包含时间序列和各项指标数据
    """
    # 复制全局数据以避免修改原始数据
    filtered_df = GLOBAL_DATA.copy()
    
    print(f"筛选参数: 日期={date}, 小时={start_hour}-{end_hour}, 分钟={start_minute}-{end_minute}")
    print(f"原始数据量: {len(filtered_df)} 条记录")
    
    # 按日期筛选
    if date:
        print(f"数据中的日期样例: {sorted(filtered_df['日期'].unique())[:10]}")
        
        # 处理日期格式
        if isinstance(date, str) and len(date) == 10:
            target_date = date
        else:
            try:
                target_date = pd.to_datetime(date).strftime('%Y-%m-%d')
            except:
                target_date = date
        
        print(f"目标日期: {target_date}")
        print(f"目标日期是否在数据中: {target_date in filtered_df['日期'].values}")
        
        # 执行日期筛选
        filtered_df = filtered_df[filtered_df['日期'] == target_date]
        print(f"按日期筛选后: {len(filtered_df)} 条记录")
        
        # 如果日期筛选后没有数据，直接返回空结果
        if len(filtered_df) == 0:
            print("日期筛选后没有数据，返回空结果")
            return {
                'time': [],
                'time_formatted': [],
                '8井油压': [],
                '8井套压': [],
                '9井压力': [],
                '9井流量': [],
                '8/9井压力差值': []
            }
    
    # 显示当前数据的时间范围（用于调试）
    if len(filtered_df) > 0:
        hour_range = f"{filtered_df['小时'].min()}-{filtered_df['小时'].max()}"
        minute_range = f"{filtered_df['分钟'].min()}-{filtered_df['分钟'].max()}"
        
        sample_times = filtered_df[['小时', '分钟']].head(10)
    
    # 按时间范围筛选（如果提供了完整的时间参数）
    if start_hour is not None and end_hour is not None and start_minute is not None and end_minute is not None:
        # 将时间转换为总分钟数以便比较
        start_total_minutes = start_hour * 60 + start_minute
        end_total_minutes = end_hour * 60 + end_minute
        
        filtered_df['总分钟'] = filtered_df['小时'] * 60 + filtered_df['分钟']
        
        if len(filtered_df) > 0:
            total_min_range = f"{filtered_df['总分钟'].min()}-{filtered_df['总分钟'].max()}"
        
        # 处理跨日时间范围（如23:30到01:30）
        if start_total_minutes <= end_total_minutes:
            # 正常时间范围
            time_condition = ((filtered_df['总分钟'] >= start_total_minutes) & 
                            (filtered_df['总分钟'] <= end_total_minutes))
        else:
            # 跨日时间范围
            time_condition = ((filtered_df['总分钟'] >= start_total_minutes) | 
                            (filtered_df['总分钟'] <= end_total_minutes))
        
        filtered_df = filtered_df[time_condition]
        filtered_df = filtered_df.drop('总分钟', axis=1)  # 删除临时列
        print(f"时间筛选后: {len(filtered_df)} 条记录")
    
    # 部分时间参数筛选（如果只提供了部分时间参数）
    elif start_hour is not None or end_hour is not None or start_minute is not None or end_minute is not None:
        print("执行部分时间筛选")
        
        # 按开始小时筛选
        if start_hour is not None:
            filtered_df = filtered_df[filtered_df['小时'] >= start_hour]
            print(f"开始小时筛选后: {len(filtered_df)} 条记录")
            
        # 按结束小时筛选
        if end_hour is not None:
            filtered_df = filtered_df[filtered_df['小时'] <= end_hour]
            print(f"结束小时筛选后: {len(filtered_df)} 条记录")
            
        # 按开始分钟筛选
        if start_minute is not None:
            filtered_df = filtered_df[filtered_df['分钟'] >= start_minute]
            print(f"开始分钟筛选后: {len(filtered_df)} 条记录")
            
        # 按结束分钟筛选
        if end_minute is not None:
            filtered_df = filtered_df[filtered_df['分钟'] <= end_minute]
            print(f"结束分钟筛选后: {len(filtered_df)} 条记录")
    
    # 如果筛选后没有数据，返回空结果
    if len(filtered_df) == 0:
        print("筛选后没有数据")
        return {
            'time': [],
            'time_formatted': [],
            '8井油压': [],
            '8井套压': [],
            '9井压力': [],
            '9井流量': [],
            '8/9井压力差值': []
        }
    
    # 将秒序号转换为时:分:秒格式
    def seconds_to_time_format(seconds, date):
        # 根据日期确定起始时间偏移量
        if date == '2021-03-14':
            # 2021-3-14从16:00:00开始，即从57600秒开始（16*3600）
            start_offset = 16 * 3600
        else:
            # 其他日期从0:00:00开始
            start_offset = 0
        
        # 计算总秒数
        total_seconds = int(seconds) + start_offset
        
        # 计算小时、分钟、秒
        hours = (total_seconds // 3600) % 24  # 取模24确保在0-23范围内
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    # 生成格式化的时间列表
    time_formatted = [seconds_to_time_format(sec, target_date) for sec in filtered_df['每日对应秒序号'].tolist()]
    
    # 构建返回结果
    result = {
        'time': filtered_df['每日对应秒序号'].tolist(),  # 保留原始秒序号用于内部计算
        'time_formatted': time_formatted,  # 新增格式化时间用于显示
        '8井油压': filtered_df['8井油压'].tolist(),
        '8井套压': filtered_df['8井套压'].tolist(),
        '9井压力': filtered_df['9井压力'].tolist(),
        '9井流量': filtered_df['9井流量'].tolist(),
        '8/9井压力差值': filtered_df['8/9井压力差值'].tolist()
    }
    
    print(f"返回数据点数: {len(result['time'])}")
    return result

@app.route('/')
def index():
    """
    主页路由
    
    渲染主页模板，如果数据未加载则先加载数据。
    
    Returns:
        str: 渲染后的HTML页面
    """
    if GLOBAL_DATA is None:
        load_data_to_memory()
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    """
    获取筛选后的数据API接口
    
    根据URL参数筛选数据并返回JSON格式的结果。
    支持的参数：
    - date: 日期 (YYYY-MM-DD)
    - start_hour: 开始小时
    - end_hour: 结束小时
    - start_minute: 开始分钟
    - end_minute: 结束分钟
    
    Returns:
        Response: JSON格式的数据或错误信息
    """
    try:
        # 确保数据已加载
        if GLOBAL_DATA is None:
            if not load_data_to_memory():
                return jsonify({'error': '数据加载失败，请检查数据文件'})
            
        # 获取URL参数
        date = request.args.get('date')
        start_hour = request.args.get('start_hour')
        end_hour = request.args.get('end_hour')
        start_minute = request.args.get('start_minute')
        end_minute = request.args.get('end_minute')
        
        print(f"原始参数: date={date}, start_hour={start_hour}, end_hour={end_hour}, start_minute={start_minute}, end_minute={end_minute}")
        
        # 转换参数类型
        start_hour = int(start_hour) if start_hour else None
        end_hour = int(end_hour) if end_hour else None
        start_minute = int(start_minute) if start_minute else None
        end_minute = int(end_minute) if end_minute else None
        
        # 获取筛选后的数据
        result = get_filtered_data(date, start_hour, end_hour, start_minute, end_minute)
        
        # 处理NaN和无穷大值，将其转换为null
        import math
        for key, values in result.items():
            if isinstance(values, list):
                result[key] = [None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v for v in values]
        
        return jsonify(result)
    except Exception as e:
        print(f"获取数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/get_date_range')
def get_date_range():
    """
    获取可用日期范围API接口
    
    返回数据中所有可用的日期列表，用于前端日期选择器。
    
    Returns:
        Response: JSON格式的日期列表或错误信息
    """
    try:
        # 确保数据已加载
        if DATE_RANGE is None:
            if not load_data_to_memory():
                return jsonify({'error': '数据加载失败，请检查数据文件'})
        
        print(f"返回日期范围: {len(DATE_RANGE)} 个日期")
        return jsonify({'dates': DATE_RANGE})
    except Exception as e:
        print(f"获取日期范围时出错: {e}")
        return jsonify({'error': str(e)})

@app.route('/debug_data')
def debug_data():
    """
    数据调试信息API接口
    
    返回数据的基本信息，用于调试和问题排查。
    包括记录总数、列名、唯一日期等信息。
    
    Returns:
        Response: JSON格式的调试信息
    """
    try:
        if GLOBAL_DATA is None:
            return jsonify({'error': '数据未加载'})
        
        # 收集调试信息
        debug_info = {
            'total_records': len(GLOBAL_DATA),
            'columns': GLOBAL_DATA.columns.tolist(),
            'unique_dates': sorted(GLOBAL_DATA['日期'].unique().tolist()),
            'date_column_sample': GLOBAL_DATA['日期'].head(10).tolist(),
            'date_column_dtype': str(GLOBAL_DATA['日期'].dtype)
        }
        
        # 检查特定日期的数据量
        target_dates = ['2021-03-14', '2021-03-15', '2021-03-16', '2021-03-17']
        for target_date in target_dates:
            count = len(GLOBAL_DATA[GLOBAL_DATA['日期'] == target_date])
            debug_info[f'count_{target_date}'] = count
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({'error': str(e)})

def cleanup():
    """
    应用关闭时的清理工作
    
    释放内存中的数据，确保应用优雅关闭。
    """
    global GLOBAL_DATA, DATE_RANGE
    GLOBAL_DATA = None
    DATE_RANGE = None
    print("应用关闭，内存已清理")

# 注册应用关闭时的清理函数
atexit.register(cleanup)

# 应用入口点
if __name__ == '__main__':

    print("启动应用...")
    load_data_to_memory()
    app.run(debug=True, port=6006)