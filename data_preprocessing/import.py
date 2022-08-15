import base64
import json
import time
import uuid
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
# 调用解码
from matplotlib import pyplot as plt
from vmdpy import VMD


def filter(s):
    alpha, tau, K, DC, init, tol = 2000.0, 0, 5, 0, 1, 1e-7
    (u, u_hat, omega) = VMD(s, alpha, tau, K, DC, init, tol)
    y = u[0] + u[1] + u[2]
    if len(y) != len(s):
        y = np.insert(y, 0, y[0:len(s) - len(y)])
    return y


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
    url_suffix = ["/cgmdc-data-api-service/api/patientDataApi/patientGlucoseDataList",  # CGM接口
                  "/cgmdc-data-api-service/api/patientDataApi/diningActionList",  # carb接口
                  "/cgmdc-data-api-service/api/patientDataApi/drugActionList"]  # insulin 接口
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


# 获取请求-CGM佩戴周期数据
def getCGMPeriodFromInterface(user_id):
    """
    Arguments :
        pageIndex : int.
        pageSize : int.
    """
    url_suffix = "/cgmdc-data-api-service/api/deviceDataApi/patientInfoPage"
    url = "https://cgmdc-api.si-datacenter.com" + url_suffix

    head = {'akid': '1d8d659dcad140aba7d329158f1cb756',
            'mid': aes_encrypt(str(uuid.uuid1())),
            'ts': aes_encrypt(time.strftime("%Y%m%d%H%M%S"))}

    body = {
        "patientId": user_id,
        "macAddress": "",
        "dataSourceSysFlag": 1
    }

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
    insulin_columns = ['actionTime', 'drugTime', 'actionType', 'dose', 'infusionMode', 'conventionalWaveDose',
                       'squareWaveDose',
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
    carb_columns = ['actionTime', 'diningType', 'foodName', 'weight', 'weightUnit', 'carbonWaterWeight', 'digestType',
                    'curveFeature', 'digestType']
    carb = pd.DataFrame(columns=carb_columns)
    for index, item in enumerate(res_data['data']['data']):
        carb.loc[index] = itemgetter(*carb_columns)(item)
    carb['Time_stamp'] = pd.to_datetime(carb['actionTime']).astype('int64')
    carb['actionTime'] = pd.to_datetime(carb['actionTime'])
    carb = carb[['Time_stamp', *carb_columns]]
    carb.sort_values(by='actionTime', ascending=True, inplace=True)
    return cgm, carb, insulin


# 导出泵基础率数据
def getBasal(user_id):
    res_data = getPumpBasalFromInterface(user_id)
    basal = []
    for item in res_data['data']:
        basal.append(item['dose'])
    return basal


# 导出CGM佩戴周期数据
def getCGMPeriodList(user_id):
    res_data = getCGMPeriodFromInterface(user_id)
    cgm_columns = ['macAddress', 'startTime', 'endTime']
    cgm_list = pd.DataFrame(columns=cgm_columns)
    for index, item in enumerate(res_data['data']):
        cgm_list.loc[index] = itemgetter(*cgm_columns)(item)

    # cgm_list['startTime'] = pd.to_datetime(cgm_list['startTime'])
    # cgm_list['endTime'] = pd.to_datetime(cgm_list['endTime'])

    return cgm_list


# 病人类
class Patient:
    def __init__(self, patientID, CGMNUM, food_name):
        self.ID = patientID
        self.CGMNUM = CGMNUM
        self.CGM = pd.DataFrame()
        self.Carb = pd.DataFrame()
        self.Insulin = pd.DataFrame()
        self.CarbProblem = pd.DataFrame()
        self.InsulinProblem = pd.DataFrame()
        self.CarbInsulinTable = pd.DataFrame()
        self.MvpAverageModelData = pd.DataFrame()
        self.MvpMealModelData = pd.DataFrame()
        self.OLSData = pd.DataFrame()
        self.Meal = [5, 10, 10, 15, 17, 23]
        self.MaxCarb = 300
        self.MaxInsulin = 20
        self.CGMPeriod = pd.DataFrame()
        self.food_name = food_name
        # 初始化
        self.getCGMPeriod()

        for idx in range(self.CGMNUM):
            star_time = self.CGMPeriod.iloc[idx, 0].split(" ")[0]
            end_time = self.CGMPeriod.iloc[idx, 1].split(" ")[0]
            cgm_file = Path(f'data/InitializationData/CGM_{self.ID}_{star_time}_{end_time}.csv')

            if cgm_file.is_file():
                self.readInitializationData(star_time, end_time)
            else:
                self.getPeriodPatientData(star_time, end_time)
                self.saveInitializationData(star_time, end_time)

            self.getMvpMealModelData()
            self.transMvpMealModelData()


    def saveInitializationData(self, star_time, end_time):
        self.CGM.to_csv(
            f'data/InitializationData/CGM_{self.ID}_{star_time}_{end_time}.csv')
        self.Carb.to_csv(
            f'data/InitializationData/Carb_{self.ID}_{star_time}_{end_time}.csv')
        self.Insulin.to_csv(
            f'data/InitializationData/Insulin_{self.ID}_{star_time}_{end_time}.csv')

    def readInitializationData(self, star_time, end_time):
        self.CGM = pd.read_csv(f'data/InitializationData/CGM_{self.ID}_{star_time}_{end_time}.csv')
        self.Carb = pd.read_csv(f'data/InitializationData/Carb_{self.ID}_{star_time}_{end_time}.csv')
        self.Insulin = pd.read_csv(
            f'data/InitializationData/Insulin_{self.ID}_{star_time}_{end_time}.csv')

        self.CGM['glucoseTime'] = pd.to_datetime(self.CGM['glucoseTime'])
        self.Carb['actionTime'] = pd.to_datetime(self.Carb['actionTime'])
        self.Insulin['actionTime'] = pd.to_datetime(self.Insulin['actionTime'])

    def getCGMPeriod(self):
        CGMList = getCGMPeriodList(self.ID)
        self.CGMPeriod = CGMList.iloc[-(self.CGMNUM + 1):, 1:3]

    def getPeriodPatientData(self, star_time, end_time):
        self.CGM, self.Carb, self.Insulin = getPeriodData(self.ID, star_time, end_time)

    def getCarbProblem(self):
        carb = self.Carb.copy()
        carb['problemID'] = 0
        carb['hour'] = carb['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        # 查询餐食类型与时间不匹配记录
        carb.iloc[carb.query(f"diningType==1 and hour > {self.Meal[1]}").index, -1] = 1
        carb.iloc[carb.query(f"diningType==2 and hour > {self.Meal[3]} and hour < {self.Meal[2]} ").index, -1] = 1
        carb.iloc[carb.query(f"diningType==3 and hour < {self.Meal[4]}").index, -1] = 1

        # 查看碳水异常数据
        carb.iloc[carb.query("carbonWaterWeight<=0").index, -1] = 2
        carb.iloc[carb.query(f"carbonWaterWeight>{self.MaxCarb}").index, -1] = 2

        # 查看碳水异常数据
        self.CarbProblem = carb.query('problemID == 1 or problemID == 2')[
            ['actionTime', 'diningType', 'foodName', 'weight', 'carbonWaterWeight', 'problemID']].copy()
        self.CarbProblem.reset_index(drop=True)

    def getInsulinProblem(self):
        insulin = self.Insulin.copy()
        insulin['problemID'] = 0
        insulin['hour'] = insulin['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        # 查询用药类型与时间不匹配记录
        insulin.iloc[insulin.query(f"drugTime==1 and {self.Meal[1]}").index, -1] = 1
        insulin.iloc[insulin.query(f"drugTime==2 and hour > {self.Meal[3]} and hour < {self.Meal[2]} ").index, -1] = 1
        insulin.iloc[insulin.query(f"drugTime==3 and hour < {self.Meal[4]}").index, -1] = 1

        # 查看用药异常数据
        insulin.iloc[insulin.query("dose <= 0").index, -1] = 2
        insulin.iloc[insulin.query(f"dose > {self.MaxInsulin}").index, -1] = 2
        insulin.iloc[insulin.query("infusionMode != 1 and (squareWaveDose == 0 or squareWaveTime == 0)").index, -1] = 2

        self.InsulinProblem = insulin.query('problemID == 1 or problemID == 2')[
            ['actionTime', 'drugTime', 'actionType', 'dose', 'infusionMode', 'conventionalWaveDose', 'squareWaveDose',
             'squareWaveTime']].copy()
        self.InsulinProblem.reset_index(drop=True)

    def getCarbInsulinTable(self):
        carb = self.Carb.copy()
        insulin = self.Insulin.copy()
        cgm = self.CGM.copy()

        # 碳水
        carb['hour'] = carb['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        carb['day'] = carb['actionTime'].apply(lambda x: x.month * 100 + x.day)

        # 每天打卡情况
        carb_day = carb['diningType'].groupby(carb['day']).value_counts().unstack().copy()
        carb_day['carb1'] = carb.query('diningType==1')[['carbonWaterWeight', 'day']].groupby('day').sum()
        carb_day['carb2'] = carb.query('diningType==2')[['carbonWaterWeight', 'day']].groupby('day').sum()
        carb_day['carb3'] = carb.query('diningType==3')[['carbonWaterWeight', 'day']].groupby('day').sum()
        carb_day['carb4'] = carb.query('diningType==4')[['carbonWaterWeight', 'day']].groupby('day').sum()

        # if 4 not in carb_day.columns:
        #     carb_day[4] = np.NaN

        carb_day.rename(columns={1: 'carb_num1', 2: 'carb_num2', 3: 'carb_num3', 4: 'carb_num4'}, inplace=True)
        # carb_day = carb_day[['carb_num1', 'carb_num2', 'carb_num3', 'carb_num4', 'carb1', 'carb2', 'carb3', 'carb4']]

        carb_day['total_day_carb'] = carb['carbonWaterWeight'].groupby(carb['day']).sum()

        # 胰岛素   
        insulin['hour'] = insulin['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        insulin['day'] = insulin['actionTime'].apply(lambda x: x.month * 100 + x.day)

        # 每天打卡情况
        insulin_day = insulin['drugTime'].groupby(insulin['day']).value_counts().unstack().copy()
        insulin_day['insulin1'] = insulin.query('drugTime==1')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin2'] = insulin.query('drugTime==2')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin3'] = insulin.query('drugTime==3')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin4'] = insulin.query('drugTime==4')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin5'] = insulin.query('drugTime==5')['dose'].groupby(insulin['day']).sum()
        insulin_day['insulin6'] = insulin.query('drugTime==6')['dose'].groupby(insulin['day']).sum()

        # if 4 not in insulin_day.columns:
        #     insulin_day[4] = np.NaN
        # if 5 not in insulin_day.columns:
        #     insulin_day[5] = np.NaN
        # if 6 not in insulin_day.columns:
        #     insulin_day[6] = np.NaN

        insulin_day.rename(
            columns={1: 'insulin_num1', 2: 'insulin_num2', 3: 'insulin_num3', 4: 'insulin_num4', 5: 'insulin_num5',
                     6: 'insulin_num6'}, inplace=True)
        # insulin_day = insulin_day[
        #     ['insulin_num1', 'insulin_num2', 'insulin_num3', 'insulin_num4', 'insulin_num5', 'insulin_num6', 'insulin1',
        #      'insulin2', 'insulin3', 'insulin4', 'insulin5', 'insulin6']].copy()

        insulin_day['total_day_insulin'] = insulin['dose'].groupby(insulin['day']).sum().copy()

        # 碳水胰岛素
        carb_insulin_day = pd.concat([carb_day, insulin_day], axis=1)
        carb_insulin_day['IC1'] = (carb_day['carb1'] / insulin_day['insulin1'])
        carb_insulin_day['IC2'] = (carb_day['carb2'] / insulin_day['insulin2'])
        carb_insulin_day['IC3'] = (carb_day['carb3'] / insulin_day['insulin3'])
        carb_insulin_day['IC_day'] = (carb_day['total_day_carb'] / insulin_day['total_day_insulin'])

        # CGM
        cgm['day'] = cgm['glucoseTime'].apply(lambda x: x.month * 100 + x.day)
        cgm['hour'] = cgm['glucoseTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        cgm_day = pd.DataFrame()

        cgm_day['cgm_mean1'] = cgm.query(f'hour > {self.Meal[0]} and hour < {self.Meal[1]}').groupby('day')[
            'glucoseValue'].mean()
        cgm_day['cgm_mean2'] = cgm.query(f'hour > {self.Meal[2]} and hour < {self.Meal[3]}').groupby('day')[
            'glucoseValue'].mean()
        cgm_day['cgm_mean3'] = cgm.query(f'hour > {self.Meal[4]} and hour < {self.Meal[5]}').groupby('day')[
            'glucoseValue'].mean()
        cgm_day['cgm_day_mean'] = cgm.query(f'hour > {self.Meal[0]} and hour < {self.Meal[5]}').groupby('day')[
            'glucoseValue'].mean()

        # 一览表
        carb_insulin_day[['cgm_mean1', 'cgm_mean2', 'cgm_mean3', 'cgm_day_mean']] = cgm_day

        self.CarbInsulinTable = carb_insulin_day.copy()

    def getMvpAverageModelData(self):
        carb = self.Carb.copy()
        insulin = self.Insulin.copy()
        cgm = self.CGM.copy()

        # 提取碳水信息表
        carb['hour'] = carb['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        carb['day'] = carb['actionTime'].apply(lambda x: x.month * 100 + x.day)
        carb_table = pd.DataFrame()
        carb_table['time_stamp'] = carb['Time_stamp'].copy()
        carb_table['name'] = carb['foodName'].copy()
        carb_table['carb'] = carb['carbonWaterWeight'].copy()
        carb_table['digestType'] = carb['digestType'].copy()
        carb_table = carb_table.T.drop_duplicates().T

        # 提取胰岛素信息表
        insulin['hour'] = insulin['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        insulin['day'] = insulin['actionTime'].apply(lambda x: x.month * 100 + x.day)
        insulin_table = pd.DataFrame()
        insulin_table['time_stamp'] = insulin['Time_stamp'].copy()
        insulin_table['dose'] = insulin['dose'].copy()
        insulin_table['infusionMode'] = insulin['infusionMode'].copy()
        insulin_table['conventionalWaveDose'] = insulin['conventionalWaveDose'].copy()
        insulin_table['squareWaveDose'] = insulin['squareWaveDose'].copy()
        insulin_table['squareWaveTime'] = insulin['squareWaveTime'].copy()
        insulin_table = insulin_table.T.drop_duplicates().T

        # 连接表
        carb_insulin_table = pd.merge(carb_table, insulin_table, how='outer', left_on='time_stamp',
                                      right_on='time_stamp')
        carb_insulin_table.fillna(0, inplace=True)
        carb_insulin_table = carb_insulin_table.sort_values(by='time_stamp', ascending=True)
        carb_insulin_table.reset_index(drop=True, inplace=True)
        carb_insulin_table['time'] = pd.to_datetime(carb_insulin_table['time_stamp'], unit='ns')

        # CGM
        cgm_table = pd.DataFrame()
        cgm_table['time_stamp'] = cgm['Time_stamp'].copy()
        cgm_table['bg'] = cgm['glucoseValue'].copy()

        # 定义导出表
        data_table = pd.merge(cgm_table, carb_insulin_table, how='outer', left_on='time_stamp', right_on='time_stamp')
        data_table.fillna(0, inplace=True)
        data_table = data_table.sort_values(by='time_stamp', ascending=True)
        data_table.reset_index(drop=True, inplace=True)
        data_table['time'] = pd.to_datetime(data_table['time_stamp'], unit='ns')
        data_table['day'] = data_table['time'].apply(lambda x: x.month * 100 + x.day)

        self.MvpAverageModelData = data_table[
            ['time', 'time_stamp', 'bg', 'name', 'carb', 'dose', 'squareWaveDose', 'squareWaveTime']].copy()

    def getMvpMealModelData(self):
        carb = self.Carb.copy()
        insulin = self.Insulin.copy()
        cgm = self.CGM.copy()

        # 提取碳水信息表
        carb['problemID'] = 0
        carb['hour'] = carb['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        carb['day'] = carb['actionTime'].apply(lambda x: x.month * 100 + x.day)

        # 查询餐食类型与时间不匹配记录
        carb.iloc[carb.query(f"diningType==1 and hour > {self.Meal[1]}").index, -1] = 1
        carb.iloc[carb.query(f"diningType==2 and hour > {self.Meal[3]} and hour < {self.Meal[2]} ").index, -1] = 1
        carb.iloc[carb.query(f"diningType==3 and hour < {self.Meal[4]}").index, -1] = 1

        # 查看碳水异常数据
        carb.iloc[carb.query("carbonWaterWeight<=0").index, -1] = 2
        carb.iloc[carb.query(f"carbonWaterWeight>{self.MaxCarb}").index, -1] = 2

        # 剔除碳水异常数据
        carb = carb.query('problemID == 0')[
            ['Time_stamp', 'actionTime', 'diningType', 'foodName', 'weight', 'carbonWaterWeight', 'problemID']].copy()

        # 提取碳水信息表
        carb_table = pd.DataFrame()
        carb_table['time_stamp'] = carb['Time_stamp'].copy()
        carb_table['name'] = carb['foodName'].copy()
        carb_table['carb'] = carb['carbonWaterWeight'].copy()
        carb_table['diningType'] = carb['diningType'].copy()
        carb_table = carb_table.T.drop_duplicates().T

        # 提取胰岛素信息表
        insulin['problemID'] = 0
        insulin['hour'] = insulin['actionTime'].apply(lambda x: round(x.hour + x.minute / 60, 2))
        insulin['day'] = insulin['actionTime'].apply(lambda x: x.month * 100 + x.day)

        # 查询用药类型与时间不匹配记录
        insulin.iloc[insulin.query(f"drugTime==1 and hour > {self.Meal[1]}").index, -1] = 1
        insulin.iloc[insulin.query(f"drugTime==2 and hour > {self.Meal[3]} and hour < {self.Meal[2]} ").index, -1] = 1
        insulin.iloc[insulin.query(f"drugTime==3 and hour < {self.Meal[4]}").index, -1] = 1

        # 查看用药异常数据
        insulin.iloc[insulin.query("dose <= 0").index, -1] = 2
        insulin.iloc[insulin.query(f"dose > {self.MaxInsulin}").index, -1] = 2
        insulin.iloc[insulin.query("infusionMode != 1 and (squareWaveDose == 0 or squareWaveTime == 0)").index, -1] = 2

        insulin = insulin.query('problemID == 0')[
            ['Time_stamp', 'actionTime', 'drugTime', 'actionType', 'dose', 'infusionMode', 'conventionalWaveDose',
             'squareWaveDose', 'squareWaveTime']].copy()

        insulin_table = pd.DataFrame()
        insulin_table['time_stamp'] = insulin['Time_stamp'].copy()
        insulin_table['dose'] = insulin['dose'].copy()
        insulin_table['infusionMode'] = insulin['infusionMode'].copy()
        insulin_table['conventionalWaveDose'] = insulin['conventionalWaveDose'].copy()
        insulin_table['squareWaveDose'] = insulin['squareWaveDose'].copy()
        insulin_table['squareWaveTime'] = insulin['squareWaveTime'].copy()
        insulin_table = insulin_table.T.drop_duplicates().T

        # 连接表
        carb_insulin_table = pd.merge(carb_table, insulin_table, how='outer', left_on='time_stamp',
                                      right_on='time_stamp')
        carb_insulin_table.fillna(0, inplace=True)
        carb_insulin_table = carb_insulin_table.sort_values(by='time_stamp', ascending=True)
        carb_insulin_table.reset_index(drop=True, inplace=True)
        carb_insulin_table['time'] = pd.to_datetime(carb_insulin_table['time_stamp'], unit='ns')

        # CGM
        cgm_table = pd.DataFrame()
        cgm_table['time_stamp'] = cgm['Time_stamp'].copy()
        cgm_table['bg'] = cgm['glucoseValue'].copy()

        # 定义导出表
        data_table = pd.merge(cgm_table, carb_insulin_table, how='outer', left_on='time_stamp', right_on='time_stamp')
        data_table.fillna(0, inplace=True)
        data_table = data_table.sort_values(by='time_stamp', ascending=True)
        data_table.reset_index(drop=True, inplace=True)
        data_table['time'] = pd.to_datetime(data_table['time_stamp'], unit='ns')
        data_table['day'] = data_table['time'].apply(lambda x: x.month * 100 + x.day)

        self.MvpMealModelData = data_table[
            ['time', 'time_stamp', 'bg', 'name', 'carb', 'dose', 'infusionMode', 'squareWaveDose',
             'squareWaveTime']].copy()

    def getOLSData(self):
        OW = 24 * 60
        step = 30
        star_tix, end_tix, index_num = 0, 0, 0
        cgm_mean, carb_sum, insulin_sum = [], [], []

        while end_tix < self.MvpAverageModelData.shape[0]:
            star_tix = step * index_num
            end_tix = step * (index_num + 1) + OW
            cgm_mean.append(self.MvpAverageModelData['bg'][star_tix:end_tix].mean())
            carb_sum.append(self.MvpAverageModelData['carb'][star_tix:end_tix].sum())
            insulin_sum.append(self.MvpAverageModelData['dose'][star_tix:end_tix].sum())
            index_num += 1
            # print(index_num)

        self.OLSData['cgm'] = cgm_mean
        self.OLSData['carb'] = carb_sum
        self.OLSData['insulin'] = insulin_sum

    def OLS(self):
        # 相关性
        X = self.OLSData[['carb', 'insulin']]
        X = sm.add_constant(X)
        Y = self.OLSData['cgm']
        model = sm.OLS(Y, X).fit()
        # predictions = model.predict(X)
        print(model.summary())

        # from statsmodels.sandbox.regression.predstd import wls_prediction_std
        # _, lower, upper = wls_prediction_std(model)
        # x = self.OLSData['carb']
        # y = self.OLSData['insulin']
        # z = self.OLSData['cgm']

        fig1 = plt.figure(figsize=(15, 8))
        fig1 = sm.graphics.plot_regress_exog(model, 'carb', fig=fig1)
        fig1.show()

        fig2 = plt.figure(figsize=(15, 8))
        fig2 = sm.graphics.plot_regress_exog(model, 'insulin', fig=fig2)
        fig2.show()

    def transMvpMealModelData(self):
        transMvpMealData = pd.DataFrame()
        transMvpMealData = self.MvpMealModelData.copy()

        # 基础率
        basal = getBasal(self.ID)
        transMvpMealData['day'] = transMvpMealData['time'].apply(lambda x: x.month * 100 + x.day)
        transMvpMealData['hour'] = transMvpMealData['time'].apply(lambda x: x.hour)
        transMvpMealData['basal'] = transMvpMealData['hour'].apply(lambda x: basal[x - 1] / 60)

        # bg滤波
        transMvpMealData['filted_bg'] = filter(transMvpMealData['bg'])

        # 方波转化
        transMvpMealData['step_dose'] = 0
        transMvpMealData.reset_index(drop=True, inplace=True)

        # 查找方波
        squareWaveTime = transMvpMealData.query('infusionMode==3')['squareWaveTime'].values
        squareWaveDose = transMvpMealData.query('infusionMode==3')['squareWaveDose'].values
        squareWaveinex = transMvpMealData.query('infusionMode==3')['squareWaveDose'].index

        tem_step_dose = transMvpMealData['step_dose'].copy()

        if any(squareWaveTime):
            for id, idx in enumerate(squareWaveinex):
                tem_step_dose[idx:idx + int(squareWaveTime[id])] = transMvpMealData['step_dose'][
                                                                   idx:idx + int(squareWaveTime[id])].apply(
                    lambda x: x + squareWaveDose[id] / squareWaveTime[id])
        transMvpMealData['step_dose'] = tem_step_dose

        transMvpMealData['dose'] = transMvpMealData['dose'] - transMvpMealData['squareWaveDose']
        transMvpMealData['dose'] = transMvpMealData['dose'] + transMvpMealData['step_dose']

        # transMvpMealData[['time', 'time_stamp', 'bg', 'filted_bg', 'carb', 'dose', 'basal']].to_csv(
        #     f'model/mvpdata/{user_id}_{it}.csv', index=None)

        transMvpMealData['dose'] = transMvpMealData['dose'] + transMvpMealData['basal']

        transMvpMealData.reset_index(drop=True, inplace=True)

        # 食物筛选
        transMvpMealData['food_name'] = transMvpMealData['name'].str.find(self.food_name)
        food_index = transMvpMealData.query('food_name==0').index
        keep_len = 2

        food_data = pd.DataFrame()
        for jj in range(len(food_index)):
            start_ = 0 if food_index[jj] - keep_len * 60 < 0 else food_index[jj] - keep_len * 60
            end_ = transMvpMealData.shape[0] if food_index[jj] + keep_len * 60 > transMvpMealData.shape[0] else \
            food_index[
                jj] + keep_len * 60
            food_data = transMvpMealData.loc[start_:end_].copy()

            food_data['time_stamp'] = (food_data['time_stamp'] - food_data.iloc[0]['time_stamp']) / 1e9 / 60

            food_data[['time_stamp', 'bg', 'filted_bg', 'carb', 'dose', 'name']].to_csv(
                f'data/mvpdata/Meal_0_{self.ID}_{transMvpMealData["day"][start_ + 1]}_{jj:0>2d}.csv', index=None, header=None)


P = Patient(29, 2, "米饭")
# P.getMvpAverageModelData()
# P.getMvpMealModelData()
# P.transMvpMealModelData("米饭")
# P.getOLSData()
# P.OLS()
# print(P.OLSData)
# CGMList = getCGMPeriodList(29)
# a = CGMList.iloc[-2:, 1:3]
# print(a.iloc[0, 1].split(" ")[0])
# print(getCGMPeriodFromInterface("29"))
