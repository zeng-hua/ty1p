import time
import json
import uuid
import base64
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from operator import itemgetter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 调用解码
def aes_encrypt(string):
    key = 'DtwuAfLOzUB5vOFZxNcDUAts7hNJ1bH4RbjsIdtS6S8='
    aes = AES.new(base64.b64decode(key), AES.MODE_ECB)
    bytes_str = string.encode('utf-8')
    pad_bytes_str = pad(bytes_str, block_size=16)
    encrypt_str = base64.b64encode(aes.encrypt(pad_bytes_str)).decode('utf-8')

    return encrypt_str

# 获取请求-病人打卡数据
def getPatientDataFromInterface(user_id, start_date, end_date, url_id):
    """
    Arguments :
        user_id : str.
        start_date : str.
        end_date : str.
    """
    url_suffix = ["/cgmdc-data-api-service/api/patientDataApi/patientGlucoseDataList",# CGM接口
                  "/cgmdc-data-api-service/api/patientDataApi/diningActionList", # carb接口
                  "/cgmdc-data-api-service/api/patientDataApi/drugActionList"] # insulin 接口
    url = "https://cgmdc-api.si-datacenter.com" + url_suffix[url_id]

    head = {'akid': '1d8d659dcad140aba7d329158f1cb756',
            'mid': aes_encrypt(str(uuid.uuid1())),
            'ts': aes_encrypt(time.strftime("%Y%m%d%H%M%S"))}

    body = {"patientId": user_id,
            "startDate": start_date,
            "endDate": end_date,
            "sysFlag": 1}

    res = json.loads(requests.post(url=url, headers=head, json=body).text)

    return res

# 获取请求-病人列表数据
def getPatientListFromInterface(pageIndex, pageSize):
    """
    Arguments :
        pageIndex : int.
        pageSize : int.
    """
    url_suffix = ["/cgmdc-data-api-service/api/patientDataApi/patientInfoPage",
                  "/cgmdc-data-api-service/api/patientDataApi/patientDataList"]
    url = "https://cgmdc-api.si-datacenter.com" + url_suffix[0]

    head = {'akid': '1d8d659dcad140aba7d329158f1cb756',
            'mid': aes_encrypt(str(uuid.uuid1())),
            'ts': aes_encrypt(time.strftime("%Y%m%d%H%M%S"))}

    body = {
        "pageIndex": pageIndex,
        "pageSize": pageSize,
        "orderByClause": "patientId",
        "orderBy": "asc"
    }

    res = json.loads(requests.post(url=url, headers=head, json=body).text)

    return res

# 获取请求-泵基础率数据
def getPumpBasalFromInterface(user_id):
    """
    Arguments :
        pageIndex : int.
        pageSize : int.
    """
    url_suffix = "/cgmdc-data-api-service/api/medicationSchemeApi/basicSchemeList"
    url = "https://cgmdc-api.si-datacenter.com" + url_suffix

    head = {'akid': '1d8d659dcad140aba7d329158f1cb756',
        'mid': aes_encrypt(str(uuid.uuid1())),
        'ts': aes_encrypt(time.strftime("%Y%m%d%H%M%S"))}

    body = {"patientId": user_id,
        "sysFlag": 1}

    res = json.loads(requests.post(url=url, headers=head, json=body).text)

    return res


# 导出病人列表数据
def getPatientList(pageIndex, pageSize):
    res_data = getPatientListFromInterface(pageIndex, pageSize)
    patient_columns = ['patientId', 'nickname', 'sex', 'birthday', 'age', 'height', 'weight', 'breakfastTime',
                       'lunchTime', 'dinnerTime', 'getUpTime', 'bedtime']
    patientList = pd.DataFrame(columns=patient_columns)
    for index, item in enumerate(res_data['data']['records']):
        patientList.loc[index] = itemgetter(*patient_columns)(item)
    return patientList


# 导出病人打卡数据
def getPeriodData(user_id, start_date, end_date):
    # 导出cgm数据
    res_data = getPatientDataFromInterface(user_id, start_date, end_date, 0)
    cgm_columns = ['glucoseTime', 'glucoseValue']
    cgm = pd.DataFrame(columns=cgm_columns)
    for index, item in enumerate(res_data['data']['data']):
        cgm.loc[index] = itemgetter(*cgm_columns)(item)
    # cgm = cgm.query("glucoseValue != 0").copy()
    cgm['glucoseTime'] = pd.to_datetime(cgm['glucoseTime'])
    cgm = cgm.reset_index(drop=True)
    cgm['Time_stamp'] = pd.to_datetime(cgm['glucoseTime']).astype('int64')
    cgm = cgm[['Time_stamp', *cgm_columns]]
    cgm.sort_values(by='glucoseTime', ascending=True, inplace=True)
    cgm.reset_index(drop=True)

    # 导出insulin数据
    res_data = getPatientDataFromInterface(user_id, start_date, end_date, 2)
    insulin_columns = ['actionTime', 'drugTime', 'actionType', 'dose', 'infusionMode', 'conventionalWaveDose', 'squareWaveDose',
                    'squareWaveTime', 'eatTime']
    insulin = pd.DataFrame(columns=insulin_columns)
    for index, item in enumerate(res_data['data']['data']):
        insulin.loc[index] = itemgetter(*insulin_columns)(item)
    insulin['Time_stamp'] = pd.to_datetime(insulin['actionTime']).astype('int64')
    insulin['actionTime'] = pd.to_datetime(insulin['actionTime'])
    insulin = insulin[['Time_stamp', *insulin_columns]]
    insulin.sort_values(by='actionTime', ascending=True, inplace=True)

    # 导出carb数据
    res_data = getPatientDataFromInterface(user_id, start_date, end_date, 1)
    carb_columns = ['actionTime', 'diningType', 'foodName', 'weight', 'weightUnit', 'carbonWaterWeight','digestType','curveFeature','digestType']
    carb = pd.DataFrame(columns=carb_columns)
    for index, item in enumerate(res_data['data']['data']):
        carb.loc[index] = itemgetter(*carb_columns)(item)
    carb['Time_stamp'] = pd.to_datetime(carb['actionTime']).astype('int64')
    carb['actionTime'] = pd.to_datetime(carb['actionTime'])
    carb = carb[['Time_stamp', *carb_columns]]
    carb.sort_values(by='actionTime', ascending=True, inplace=True)

    return cgm,carb,insulin


# 导出泵基础率数据
def getBasal(user_id):
    res_data = getPumpBasalFromInterface(user_id)
    basal = []
    for item in res_data['data']:
        basal.append(item['dose'])
    return basal

# 病人类
class Patient:
    def __init__(self, patientID):
        self.ID = patientID
        self.CGM = pd.DataFrame()
        self.Carb = pd.DataFrame()
        self.Insulin = pd.DataFrame()
        self.CarbProblem = pd.DataFrame()
        self.InsulinProblem = pd.DataFrame()
        self.CarbInsulinTable = pd.DataFrame()
        self.Meal = [5,10,10,15,17,23]
        self.MaxCarb = 300
        self.MaxInsulin = 20 
        

    def getCGMPeriod(self):
        self.CGMPeriod = ['2022-07-11','2022-08-11']
        

    def getPeriodPatientData(self):
        self.CGM,self.Carb,self.Insulin = getPeriodData(self.ID,self.CGMPeriod[0],self.CGMPeriod[1])


    def getCarbProblem(self):
        carb = self.Carb.copy()
        carb['problemID'] = 0
        carb['hour']=carb['actionTime'].apply(lambda x : round(x.hour+x.minute/60,2))
        # 查询餐食类型与时间不匹配记录
        carb.iloc[carb.query(f"diningType==1 and hour > {self.Meal[1]}").index, -1] = 1
        carb.iloc[carb.query(f"diningType==2 and hour > {self.Meal[3]} and hour < {self.Meal[2]} ").index, -1] = 1
        carb.iloc[carb.query(f"diningType==3 and hour < {self.Meal[4]}").index, -1] = 1

        # 查看碳水异常数据
        carb.iloc[carb.query("carbonWaterWeight<=0").index, -1] = 2
        carb.iloc[carb.query(f"carbonWaterWeight>{self.MaxCarb}").index, -1] = 2
        
        # 查看碳水异常数据
        self.CarbProblem = carb.query('problemID == 1 or problemID == 2')[['actionTime','diningType','foodName','weight','carbonWaterWeight','problemID']].copy()
        self.CarbProblem.reset_index(drop=True)
        
    
    def getInsulinProblem(self):
        insulin = self.Insulin.copy()
        insulin['problemID'] = 0
        insulin['hour']=insulin['actionTime'].apply(lambda x : round(x.hour+x.minute/60,2))
        # 查询用药类型与时间不匹配记录
        insulin.iloc[insulin.query(f"drugTime==1 and {self.Meal[1]}").index, -1] = 1
        insulin.iloc[insulin.query(f"drugTime==2 and hour > {self.Meal[3]} and hour < {self.Meal[2]} ").index, -1] = 1
        insulin.iloc[insulin.query(f"drugTime==3 and hour < {self.Meal[4]}").index, -1] = 1

        # 查看用药异常数据
        insulin.iloc[insulin.query("dose <= 0").index, -1] = 2
        insulin.iloc[insulin.query(f"dose > {self.MaxInsulin}").index, -1] = 2
        insulin.iloc[insulin.query("infusionMode != 1 and (squareWaveDose == 0 or squareWaveTime == 0)").index, -1] = 2
        
        self.InsulinProblem = insulin.query('problemID == 1 or problemID == 2')[['actionTime','drugTime','actionType','dose','infusionMode','conventionalWaveDose','squareWaveDose','squareWaveTime']].copy()
        # (self.InsulinProblem).reset_index(drop=True)

    def getCarbInsulinTable(self):
        carb = self.Carb.copy()
        insulin = self.Insulin.copy()
        cgm = self.CGM.copy()
        
        # 碳水
        carb['hour']=carb['actionTime'].apply(lambda x : round(x.hour+x.minute/60,2))
        carb['day']=carb['actionTime'].apply(lambda x : x.month*100+x.day)
        
        # 每天打卡情况
        carb_day = carb['diningType'].groupby(carb['day']).value_counts().unstack()
        carb_day['carb1'] = carb.query('diningType==1')[['carbonWaterWeight','day']].groupby('day').sum()
        carb_day['carb2'] = carb.query('diningType==2')[['carbonWaterWeight','day']].groupby('day').sum()
        carb_day['carb3'] = carb.query('diningType==3')[['carbonWaterWeight','day']].groupby('day').sum()
        carb_day['carb4'] = carb.query('diningType==4')[['carbonWaterWeight','day']].groupby('day').sum()

        if 4 not in carb_day.columns:
            carb_day[4]='NaN'

        carb_day.rename(columns={1:'carb_num1',2:'carb_num2',3:'carb_num3',4:'carb_num4'}, inplace = True)
        carb_day = carb_day[['carb_num1','carb_num2','carb_num3','carb_num4','carb1','carb2','carb3','carb4']]

        carb_day['total_day_carb'] = carb['carbonWaterWeight'].groupby(carb['day']).sum()
        
        # 胰岛素   
        insulin['hour']=insulin['actionTime'].apply(lambda x : round(x.hour+x.minute/60,2))
        insulin['day']=insulin['actionTime'].apply(lambda x : x.month*100+x.day)
        
        # 每天打卡情况
        insulin_day = insulin['drugTime'].groupby(insulin['day']).value_counts().unstack()
        insulin_day['insulin1'] = insulin.query('drugTime==1')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin2'] = insulin.query('drugTime==2')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin3'] = insulin.query('drugTime==3')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin4'] = insulin.query('drugTime==4')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin5'] = insulin.query('drugTime==5')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin6'] = insulin.query('drugTime==6')['dose'].groupby(insulin['day']).sum()

        if 4 not in carb_day.columns:
            insulin_day[4]='NaN'
        if 5 not in carb_day.columns:
            insulin_day[5]='NaN'
        if 6 not in carb_day.columns:
            insulin_day[6]='NaN'

        insulin_day.rename(columns={1:'insulin_num1',2:'insulin_num2',3:'insulin_num3',4:'insulin_num4',5:'insulin_num5',6:'insulin_num6'}, inplace = True)
        insulin_day = insulin_day[['insulin_num1', 'insulin_num2', 'insulin_num3','insulin_num4', 'insulin_num5','insulin_num6','insulin1','insulin2','insulin3','insulin4','insulin5','insulin6']]

        insulin_day['total_day_insulin'] = insulin['dose'].groupby(insulin['day']).sum()
        
        # 碳水胰岛素
        carb_insulin_day = pd.concat([carb_day,insulin_day], axis=1)
        carb_insulin_day['IC1'] = (carb_day['carb1']/insulin_day['insulin1'])
        carb_insulin_day['IC2'] = (carb_day['carb2']/insulin_day['insulin2'])
        carb_insulin_day['IC3'] = (carb_day['carb3']/insulin_day['insulin3'])
        carb_insulin_day['IC_day'] = (carb_day['total_day_carb']/insulin_day['total_day_insulin'])
        
        # CGM
        cgm['day']=cgm['glucoseTime'].apply(lambda x:x.month*100+x.day)
        cgm['hour']=cgm['glucoseTime'].apply(lambda x : round(x.hour+x.minute/60,2))
        cgm_day = pd.DataFrame()

        cgm_day['cgm_mean1']= cgm.query(f'hour > {self.Meal[0]} and hour < {self.Meal[1]}').groupby('day')['glucoseValue'].mean()
        cgm_day['cgm_mean2']= cgm.query(f'hour > {self.Meal[2]} and hour < {self.Meal[3]}').groupby('day')['glucoseValue'].mean()
        cgm_day['cgm_mean3']= cgm.query(f'hour > {self.Meal[4]} and hour < {self.Meal[5]}').groupby('day')['glucoseValue'].mean()
        cgm_day['cgm_day_mean'] = cgm.query('hour > {self.Meal[0]} and hour < {self.Meal[5]}').groupby('day')['glucoseValue'].mean()
        
        # 一览表
        carb_insulin_day[['cgm_mean1','cgm_mean2','cgm_mean3','cgm_day_mean']] = cgm_day
        
        self.CarbInsulinTable = carb_insulin_day.copy()
        
        
    
P = Patient(12)
P.getPeriodPatientData()
P.getCarbInsulinTable()
self.CarbInsulinTable.to_csv('out.csv')