# -*- coding: utf-8 -*-
"""
时间范围数据筛选工具
功能：根据指定的日期和时间范围筛选数据行
"""

import pandas as pd
import os
from datetime import datetime

class TimeRangeFilter:
    """
    时间范围筛选器类
    用于根据日期和时间范围筛选数据
    """
    
    def __init__(self, data_file_path=None):
        """
        初始化筛选器
        
        Args:
            data_file_path (str): 数据文件路径，默认使用项目中的数据文件
        """
        if data_file_path is None:
            # 默认数据文件路径
            self.data_file_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "timeEvalWebData", 
                "month_3_processed.csv"
            )
        else:
            self.data_file_path = data_file_path
        
        self.data = None
        self.load_data()
    
    def load_data(self):
        """
        加载数据文件
        
        Returns:
            bool: 数据加载是否成功
        """
        try:
            if not os.path.exists(self.data_file_path):
                print(f"数据文件不存在: {self.data_file_path}")
                return False
            
            self.data = pd.read_csv(self.data_file_path)
            
            # 确保日期格式正确
            if '日期' in self.data.columns:
                self.data['日期'] = pd.to_datetime(self.data['日期']).dt.strftime('%Y-%m-%d')
            
            # 确保时间列为整数类型
            if '小时' in self.data.columns:
                self.data['小时'] = self.data['小时'].astype(int)
            if '分钟' in self.data.columns:
                self.data['分钟'] = self.data['分钟'].astype(int)
            
            print(f"数据加载成功，共 {len(self.data)} 条记录")
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def filter_by_time_range(self, start_date, start_time, end_date, end_time, output_columns=None):
        """
        根据时间范围筛选数据
        
        Args:
            start_date (str): 开始日期，格式为 'YYYY-MM-DD'
            start_time (str): 开始时间，格式为 'HH:MM'
            end_date (str): 结束日期，格式为 'YYYY-MM-DD'
            end_time (str): 结束时间，格式为 'HH:MM'
            output_columns (list): 需要输出的列名列表，默认输出所有列
        
        Returns:
            pd.DataFrame: 筛选后的数据
        """
        if self.data is None:
            print("数据未加载，请先加载数据")
            return pd.DataFrame()
        
        try:
            # 解析时间
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            
            # 验证时间格式
            if not (0 <= start_hour <= 23 and 0 <= start_minute <= 59):
                raise ValueError(f"开始时间格式错误: {start_time}")
            if not (0 <= end_hour <= 23 and 0 <= end_minute <= 59):
                raise ValueError(f"结束时间格式错误: {end_time}")
            
            # 复制数据以避免修改原始数据
            filtered_data = self.data.copy()
            
            print(f"筛选条件: {start_date} {start_time} 到 {end_date} {end_time}")
            print(f"原始数据量: {len(filtered_data)} 条记录")
            
            # 如果是同一天的时间范围
            if start_date == end_date:
                # 按日期筛选
                filtered_data = filtered_data[filtered_data['日期'] == start_date]
                print(f"按日期筛选后: {len(filtered_data)} 条记录")
                
                # 按时间范围筛选
                start_total_minutes = start_hour * 60 + start_minute
                end_total_minutes = end_hour * 60 + end_minute
                
                filtered_data['总分钟'] = filtered_data['小时'] * 60 + filtered_data['分钟']
                
                # 处理跨日时间范围（如23:30到01:30）
                if start_total_minutes <= end_total_minutes:
                    # 正常时间范围
                    time_condition = ((filtered_data['总分钟'] >= start_total_minutes) & 
                                    (filtered_data['总分钟'] <= end_total_minutes))
                else:
                    # 跨日时间范围
                    time_condition = ((filtered_data['总分钟'] >= start_total_minutes) | 
                                    (filtered_data['总分钟'] <= end_total_minutes))
                
                filtered_data = filtered_data[time_condition]
                filtered_data = filtered_data.drop('总分钟', axis=1)  # 删除临时列
                
            else:
                # 跨日期的时间范围
                start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
                end_datetime = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
                
                # 创建完整的时间戳用于比较
                filtered_data['完整时间'] = pd.to_datetime(
                    filtered_data['日期'] + ' ' + 
                    filtered_data['小时'].astype(str).str.zfill(2) + ':' + 
                    filtered_data['分钟'].astype(str).str.zfill(2)
                )
                
                # 筛选时间范围
                time_condition = ((filtered_data['完整时间'] >= start_datetime) & 
                                (filtered_data['完整时间'] <= end_datetime))
                filtered_data = filtered_data[time_condition]
                filtered_data = filtered_data.drop('完整时间', axis=1)  # 删除临时列
            
            print(f"时间筛选后: {len(filtered_data)} 条记录")
            
            # 选择输出列
            if output_columns:
                # 验证列名是否存在
                valid_columns = [col for col in output_columns if col in filtered_data.columns]
                if len(valid_columns) != len(output_columns):
                    missing_columns = set(output_columns) - set(valid_columns)
                    print(f"警告: 以下列不存在: {missing_columns}")
                filtered_data = filtered_data[valid_columns]
            
            return filtered_data
            
        except Exception as e:
            print(f"筛选过程中出错: {e}")
            return pd.DataFrame()
    
    def filter_single_day(self, date, start_time, end_time, output_columns=None):
        """
        筛选单日时间范围的数据（简化版本）
        
        Args:
            date (str): 日期，格式为 'YYYY-MM-DD'
            start_time (str): 开始时间，格式为 'HH:MM'
            end_time (str): 结束时间，格式为 'HH:MM'
            output_columns (list): 需要输出的列名列表，默认输出所有列
        
        Returns:
            pd.DataFrame: 筛选后的数据
        """
        return self.filter_by_time_range(date, start_time, date, end_time, output_columns)
    
    def get_available_dates(self):
        """
        获取数据中所有可用的日期
        
        Returns:
            list: 可用日期列表
        """
        if self.data is None or '日期' not in self.data.columns:
            return []
        
        return sorted(self.data['日期'].unique().tolist())
    
    def get_time_range_for_date(self, date):
        """
        获取指定日期的时间范围
        
        Args:
            date (str): 日期，格式为 'YYYY-MM-DD'
        
        Returns:
            tuple: (最早时间, 最晚时间) 格式为 ('HH:MM', 'HH:MM')
        """
        if self.data is None:
            return None, None
        
        date_data = self.data[self.data['日期'] == date]
        if len(date_data) == 0:
            return None, None
        
        min_hour = date_data['小时'].min()
        min_minute = date_data[date_data['小时'] == min_hour]['分钟'].min()
        
        max_hour = date_data['小时'].max()
        max_minute = date_data[date_data['小时'] == max_hour]['分钟'].max()
        
        start_time = f"{min_hour:02d}:{min_minute:02d}"
        end_time = f"{max_hour:02d}:{max_minute:02d}"
        
        return start_time, end_time
    
    def export_filtered_data(self, filtered_data, output_file_path, file_format='csv'):
        """
        导出筛选后的数据到文件
        
        Args:
            filtered_data (pd.DataFrame): 筛选后的数据
            output_file_path (str): 输出文件路径
            file_format (str): 文件格式，支持 'csv', 'excel'
        
        Returns:
            bool: 导出是否成功
        """
        try:
            if file_format.lower() == 'csv':
                filtered_data.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            elif file_format.lower() == 'excel':
                filtered_data.to_excel(output_file_path, index=False)
            else:
                print(f"不支持的文件格式: {file_format}")
                return False
            
            print(f"数据已导出到: {output_file_path}")
            return True
            
        except Exception as e:
            print(f"导出数据时出错: {e}")
            return False