import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import datetime as dt
import matplotlib.pyplot as plt
from functools import reduce
from operator import add
import os
from os.path import exists


def readData(file_path):
	sm_daily = pd.read_csv(file_path, )
	sm_doy = sm_daily.iloc[1:, 1]
	sm_daily_values = sm_daily.iloc[1:, [10, 11, 12, 13, 25, 26, 27, 28]].to_numpy().astype(float)
	sm_daily = pd.DataFrame(sm_daily_values, index=sm_doy,
							columns=['HRLDAS_ET', 'HRLDAS_0–10 cm', 'HRLDAS_10–40 cm', 'HRLDAS_40–100 cm', 'In situ_ET',
									 'In situ_0–10 cm', 'In situ_10–50 cm', 'In situ_50–100 cm'])
	return sm_daily

def AmerSitesSM(model_output,ground_measures,start, end, result_path):
	if(exists(result_path)):
		sm = pd.read_csv(result_path, index_col=0)
		sm.index = sm.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
		sm = sm.loc[start:end, :]
	else:
		sm_p0= pd.read_csv(model_output, index_col=0)
		sm_p0 = sm_p0.sort_index()
		sm_p0.index = sm_p0.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')-dt.timedelta(hours=6))# UTC to LST
		sm_p0= sm_p0.iloc[:,:4]
		sm_p0.columns=[ 'HRLDAS_5 cm', 'HRLDAS_25 cm', 'HRLDAS_70 cm','HRLDAS_150 cm']

		sm_Ne0 = pd.read_csv(ground_measures, index_col='TIMESTAMP_START', skiprows=2)
		sm_Ne0.index = sm_Ne0.index.map(lambda x: dt.datetime.strptime(str(x), '%Y%m%d%H%M'))# local standard time
		sm_Ne0 = sm_Ne0.loc[start:end,['SWC_PI_F_1_1_1','SWC_PI_F_2_1_1','SWC_PI_F_3_1_1','SWC_PI_F_1_2_1','SWC_PI_F_2_2_1','SWC_PI_F_3_2_1',
									'SWC_PI_F_1_3_1','SWC_PI_F_2_3_1','SWC_PI_F_3_3_1','SWC_PI_F_1_4_1','SWC_PI_F_2_4_1','SWC_PI_F_3_4_1']]
		sm_Ne0 = sm_Ne0.replace(-9999, np.nan)
		sm_Ne0 = sm_Ne0.dropna()
		sm_Ne0['In situ_10 cm'] = (sm_Ne0['SWC_PI_F_1_1_1'] + sm_Ne0['SWC_PI_F_2_1_1'] + sm_Ne0['SWC_PI_F_3_1_1'])/ 300
		sm_Ne0['In situ_25 cm'] = (sm_Ne0['SWC_PI_F_1_2_1'] + sm_Ne0['SWC_PI_F_2_2_1'] + sm_Ne0['SWC_PI_F_3_2_1'])/300
		sm_Ne0['In situ_50 cm'] = (sm_Ne0['SWC_PI_F_1_3_1'] + sm_Ne0['SWC_PI_F_2_3_1'] + sm_Ne0['SWC_PI_F_3_3_1']) / 300
		sm_Ne0['In situ_100 cm'] = (sm_Ne0['SWC_PI_F_1_4_1'] + sm_Ne0['SWC_PI_F_2_4_1'] + sm_Ne0['SWC_PI_F_3_4_1']) / 300
		#linear interpolation
		sm_Ne0['In situ_5 cm'] =  sm_Ne0['In situ_10 cm'] - (sm_Ne0['In situ_25 cm'] - sm_Ne0['In situ_10 cm'] )/3
		sm_Ne0['In situ_70 cm'] = sm_Ne0['In situ_50 cm'] + (sm_Ne0['In situ_100 cm'] - sm_Ne0['In situ_50 cm'])*2/5
		sm_Ne0['In situ_150 cm'] = sm_Ne0['In situ_100 cm'] + (sm_Ne0['In situ_100 cm'] - sm_Ne0['In situ_50 cm'])
		sm_Ne0 = sm_Ne0.loc[:,[ 'In situ_5 cm', 'In situ_25 cm', 'In situ_70 cm', 'In situ_150 cm']]
		#sm_Ne0['In situ_10–100 cm']= sm_Ne0['In situ_10–50 cm']-0.03
		sm = pd.merge(sm_p0, sm_Ne0, how='inner', right_index=True, left_index=True)
		sm.to_csv(result_path)
	return(sm)


def AmerSitesET(model_output, ground_measures):

	spData = pd.read_csv(model_output, index_col=0)
	spData.index = spData.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
	spData.rename(columns={'ET':'HRLDAS_ET'}, inplace=True)
	spData.iloc[0, 0] = 0
	spData = spData.astype({"HRLDAS_ET": float})
	spData= spData.iloc[:,0]



	etNe = pd.read_csv(ground_measures)
	et_day = etNe.iloc[1:,0]
	et_day = et_day.map(lambda x: dt.datetime.strptime(str(x).split('.')[0], '%Y%m%d').date())
	et_values = etNe.iloc[1:,25].to_numpy().astype(float)
	etNe =  pd.DataFrame(et_values, index=et_day,
							 columns=['In situ_ET'])
	et = pd.merge(spData, etNe, how='inner', right_index=True, left_index=True)

	return(et)

def unlSitesET(model_output, ground_measures):
	spData = pd.read_csv(model_output, index_col=0)
	spData.index = spData.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
	spData.rename(columns={'ET': 'HRLDAS_ET'}, inplace=True)
	spData = spData.iloc[:, 0]

	ET = pd.read_csv(ground_measures, index_col=0)
	ET.index = ET.index.map(lambda x: dt.datetime.strptime(x, '%m/%d/%Y'))
	et = pd.merge(ET['ET'], spData, how='inner', right_index=True, left_index=True)
	et.rename(columns={'ET': 'In situ_ET'}, inplace=True)
	return(et)

def crnSM(model_output,ground_measures, start, end, result_path):
	if(exists(result_path)):
		sm = pd.read_csv(result_path, index_col=0)
		sm.index = sm.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
		sm = sm.loc[start:end, :]
	else:
		sm_hrldas = pd.read_csv(model_output, index_col=0)
		sm_hrldas = sm_hrldas.sort_index()
		sm_hrldas.index = sm_hrldas.index.map(
			lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=6))  # UTC to LST
		sm_hrldas = sm_hrldas.iloc[:, :4]
		sm_hrldas.columns = [ 'HRLDAS_5 cm', 'HRLDAS_25 cm', 'HRLDAS_70 cm','HRLDAS_150 cm']
		sm_crn = pd.read_csv(ground_measures, sep="\s+", header=None, dtype=str)

		columnstring = "WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE T_CALC T_HR_AVG T_MAX T_MIN P_CALC SOLARAD SOLARAD_FLAG SOLARAD_MAX SOLARAD_MAX_FLAG SOLARAD_MIN SOLARAD_MIN_FLAG SUR_TEMP_TYPE SUR_TEMP SUR_TEMP_FLAG SUR_TEMP_MAX SUR_TEMP_MAX_FLAG SUR_TEMP_MIN SUR_TEMP_MIN_FLAG RH_HR_AVG RH_HR_AVG_FLAG SOIL_MOISTURE_5 SOIL_MOISTURE_10 SOIL_MOISTURE_20 SOIL_MOISTURE_50 SOIL_MOISTURE_100 SOIL_TEMP_5 SOIL_TEMP_10 SOIL_TEMP_20 SOIL_TEMP_50 SOIL_TEMP_100"
		columns =  columnstring.split(" ")
		sm_crn.columns =columns
		sm_crn.index = map(lambda d,t: dt.datetime.strptime(str(d)+str(t), '%Y%m%d%H%M'), sm_crn['LST_DATE'], sm_crn['LST_TIME'])
		sm_crn = sm_crn.loc[start:end,['SOIL_MOISTURE_5', 'SOIL_MOISTURE_10', 'SOIL_MOISTURE_20', 'SOIL_MOISTURE_50', 'SOIL_MOISTURE_100']]
		sm_crn = sm_crn.replace('-99.000',np.nan)
		sm_crn = sm_crn.dropna()
		sm_crn['In situ_5 cm'] = sm_crn['SOIL_MOISTURE_5'].to_numpy().astype(float)
		#linear interpolation
		sm_crn['In situ_25 cm'] = sm_crn['SOIL_MOISTURE_20'].to_numpy().astype(float) + \
								  (sm_crn['SOIL_MOISTURE_50'].to_numpy().astype(float)-sm_crn['SOIL_MOISTURE_20'].to_numpy().astype(float))/6
		sm_crn['In situ_70 cm'] = sm_crn['SOIL_MOISTURE_50'].to_numpy().astype(float) + \
								  (sm_crn['SOIL_MOISTURE_100'].to_numpy().astype(float)+sm_crn['SOIL_MOISTURE_50'].to_numpy().astype(float))*2/5
		sm_crn['In situ_150 cm'] = sm_crn['SOIL_MOISTURE_100'].to_numpy().astype(float) + \
								   (sm_crn['SOIL_MOISTURE_100'].to_numpy().astype(float)-sm_crn['SOIL_MOISTURE_50'].to_numpy().astype(float))

		sm_crn = sm_crn.loc[start:end,[ 'In situ_5 cm', 'In situ_25 cm', 'In situ_70 cm', 'In situ_150 cm',
									   #'SOIL_MOISTURE_5', 'SOIL_MOISTURE_10', 'SOIL_MOISTURE_20', 'SOIL_MOISTURE_50', 'SOIL_MOISTURE_100'
									   ]]

		sm = pd.merge(sm_hrldas, sm_crn, how='inner', right_index=True, left_index=True)
		#print(sm)
		sm.to_csv(result_path)
	return(sm)


def scanSM(model_output,ground_measures, start, end, result_path):
	if(exists(result_path)):
		sm = pd.read_csv(result_path, index_col=0)
		sm.index = sm.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
		sm = sm.loc[start:end,:]
	else:
		sm_hrldas = pd.read_csv(model_output, index_col=0)
		sm_hrldas = sm_hrldas.sort_index()
		sm_hrldas.index = sm_hrldas.index.map(
			lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))  # Daily sm come from 10am UTC
		sm_hrldas = sm_hrldas.iloc[:, -4:]
		sm_hrldas.columns = [ 'HRLDAS_5 cm', 'HRLDAS_25 cm', 'HRLDAS_70 cm','HRLDAS_150 cm']
		#sm_hrldas['HRLDAS_10–100 cm'] = sm_hrldas['HRLDAS_10–40 cm']

		sm_scan = pd.read_csv(ground_measures, header=60, dtype=str, index_col='Date')
		sm_scan.index = sm_scan.index.map(lambda x: dt.datetime.strptime(str(x), '%Y-%m-%d'))
		sm_scan.columns = ['SOIL_MOISTURE_5', 'SOIL_MOISTURE_10', 'SOIL_MOISTURE_20', 'SOIL_MOISTURE_50', 'SOIL_MOISTURE_100']
		sm_scan = sm_scan.dropna()
		sm_scan['In situ_5 cm'] = sm_scan['SOIL_MOISTURE_5'].to_numpy().astype(float)/100
		#linear interpolation
		sm_scan['In situ_25 cm'] = (sm_scan['SOIL_MOISTURE_20'].to_numpy().astype(float) + \
								  (sm_scan['SOIL_MOISTURE_50'].to_numpy().astype(float)-sm_scan['SOIL_MOISTURE_20'].to_numpy().astype(float))/6)/100
		sm_scan['In situ_70 cm'] = (sm_scan['SOIL_MOISTURE_50'].to_numpy().astype(float) + \
								  (sm_scan['SOIL_MOISTURE_100'].to_numpy().astype(float)+sm_scan['SOIL_MOISTURE_50'].to_numpy().astype(float))*2/5)/100
		sm_scan['In situ_150 cm'] = (sm_scan['SOIL_MOISTURE_100'].to_numpy().astype(float) + \
								   (sm_scan['SOIL_MOISTURE_100'].to_numpy().astype(float)-sm_scan['SOIL_MOISTURE_50'].to_numpy().astype(float)))/100
		sm_scan = sm_scan.loc[start:end,
				 ['In situ_5 cm', 'In situ_25 cm', 'In situ_70 cm', 'In situ_150 cm',
				  #'SOIL_MOISTURE_5', 'SOIL_MOISTURE_10', 'SOIL_MOISTURE_20', 'SOIL_MOISTURE_50', 'SOIL_MOISTURE_100'
				  ]]
		sm = pd.merge(sm_hrldas, sm_scan, how='inner', right_index=True, left_index=True)
		sm.to_csv(result_path)
	return(sm)
	#print(sm)



def mergeData():
	NE01_2020 = readData(r'../wsdata/NE01_2020.csv')
	NE02_2020 = readData(r'../wsdata/NE02_2020.csv')
	NE03_2020 = readData(r'../wsdata/NE03_2020.csv')
	NE01_2019 = readData(r'../wsdata/NE01_2019.csv')
	NE02_2019 = readData(r'../wsdata/NE03_2019.csv')
	NE03_2019 = readData(r'../wsdata/NE03_2019.csv')

	mergeD = reduce(lambda a, b: a.add(b, fill_value=0), [NE01_2019, NE02_2019, NE03_2019, NE01_2020, NE02_2020, NE03_2020])
	mergeD = mergeD.iloc[1:-1,:]

	meanD = mergeD/6
	fig = plt.figure(figsize=(20, 5))
	ax1 = fig.add_subplot(111)
	meanD[['HRLDAS_ET', 'In situ_ET']].plot(ax=ax1)
	RMSE = rmse(meanD['HRLDAS_ET'].to_numpy().astype(float), meanD['In situ_ET'].to_numpy().astype(float))
	print(RMSE)
	plt.show()

def rmse(predictions, targets):
	return np.sqrt(((predictions - targets)**2).mean())

def bias(predictions, targets):
	return (predictions-targets).mean()

def ubrmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean()-bias(predictions, targets)**2)

def r2(pred,tar):
	d1 = tar[:, np.newaxis]
	d2 = pred[:, np.newaxis]
	lrModel = LinearRegression()
	lrModel.fit(d1, d2)
	#predicts = lrModel.predict(d1)
	R2 = lrModel.score(d1, d2)
	# r2_list.append(R2)
	# print('R2 = %.2f' % R2)
	return R2


def main():
	start =  "04/01/2020"
	end = "2020103123"
	start = dt.datetime.strptime(start, '%m/%d/%Y')
	end = dt.datetime.strptime(end, '%Y%m%d%H')
	result_data = r'E:\OneDrive - George Mason University - O365 Production\work\validation\result\data/'
	model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE01_NONIRRI_2020040100.LDASOUT_DOMAIN1_SM.csv'
	ground_measures=r'E:\OneDrive - George Mason University - O365 Production\work\modelCalibration\wsdata\AMF_US-Ne1_BASE-BADM_10-5\AMF_US-Ne1_BASE_HR_10-5.csv'
	ne1_SM = AmerSitesSM(model_output,ground_measures,start, end, result_data+'AmeriFlux_NE01_2020_nonirr.csv')

	model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE02_NONIRRI_2020040100.LDASOUT_DOMAIN1_SM.csv'
	ground_measures=r'E:\OneDrive - George Mason University - O365 Production\work\modelCalibration\wsdata\AMF_US-Ne2_BASE-BADM_10-5\AMF_US-Ne2_BASE_HR_10-5.csv'
	ne2_SM = AmerSitesSM(model_output,ground_measures,start, end, result_data+'AmeriFlux_NE02_2020_nonirr.csv')

	model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE01_2020040100.LDASOUT_DOMAIN1_SM.csv'
	ground_measures=r'E:\OneDrive - George Mason University - O365 Production\work\modelCalibration\wsdata\AMF_US-Ne1_BASE-BADM_10-5\AMF_US-Ne1_BASE_HR_10-5.csv'
	ne1_SM_irri = AmerSitesSM(model_output,ground_measures,start, end, result_data+'AmeriFlux_NE01_2020_irr.csv')

	model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE02_2020040100.LDASOUT_DOMAIN1_SM.csv'
	ground_measures=r'E:\OneDrive - George Mason University - O365 Production\work\modelCalibration\wsdata\AMF_US-Ne2_BASE-BADM_10-5\AMF_US-Ne2_BASE_HR_10-5.csv'
	ne2_SM_irri = AmerSitesSM(model_output,ground_measures,start, end, result_data+'AmeriFlux_NE02_2020_irr.csv')
	# ne2_SM.to_csv(r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE02_SM.csv')

	mergeD = reduce(lambda a, b: a.add(b, fill_value=0), [ne1_SM, ne2_SM,])
	meanD = mergeD/2

	mergeD_irri = reduce(lambda a, b: a.add(b, fill_value=0), [ne1_SM_irri, ne2_SM_irri,])
	meanD_irri = mergeD_irri/2

	meanD = pd.merge(meanD, meanD_irri, how='inner', right_index=True, left_index=True, suffixes=('','_irri'))
	# model_output = r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\2020040100.LDASOUT_DOMAIN1_SM.csv'
	# ground_measures = r'F:\work\modelCalibration\wsdata\AMF_US-Ne3_BASE-BADM_10-5\AMF_US-Ne3_BASE_HR_10-5.csv'
	# ne3_SM = AmerSitesSM(model_output, ground_measures)
	# ne3_SM.to_csv(r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE03_SM.csv')

	# plt.rc('font', size=16)
	# fig = plt.figure(figsize=(20, 5))
	# ax1 = fig.add_subplot(211)
	plt.rc('font', size=12)
	fig, axs = plt.subplots(4, 1)
	meanD[['HRLDAS_5 cm_irri', 'In situ_5 cm']].plot(ax=axs[0])
	meanD[['HRLDAS_25 cm_irri', 'In situ_25 cm']].plot(ax=axs[1])
	meanD[['HRLDAS_70 cm_irri', 'In situ_70 cm']].plot(ax=axs[2])
	meanD[['HRLDAS_150 cm_irri', 'In situ_150 cm']].plot(ax=axs[3])
	# meanD[['HRLDAS_5 cm', 'HRLDAS_5 cm_irri']].plot(ax=axs[0])
	# meanD[['HRLDAS_25 cm', 'HRLDAS_25 cm_irri']].plot(ax=axs[1])
	# meanD[['HRLDAS_70 cm', 'HRLDAS_70 cm_irri']].plot(ax=axs[2])
	# meanD[['HRLDAS_150 cm', 'HRLDAS_150 cm_irri']].plot(ax=axs[3])
	plt.show()
	rmse_5 = rmse(meanD['HRLDAS_5 cm'].to_numpy().astype(float),
				  meanD['In situ_5 cm'].to_numpy().astype(float))
	rmse_25 = rmse(meanD['HRLDAS_25 cm'].to_numpy().astype(float),
				   meanD['In situ_25 cm'].to_numpy().astype(float))
	rmse_70 = rmse(meanD['HRLDAS_70 cm'].to_numpy().astype(float),
				   meanD['In situ_70 cm'].to_numpy().astype(float))
	rmse_150 = rmse(meanD['HRLDAS_150 cm'].to_numpy().astype(float),
					meanD['In situ_150 cm'].to_numpy().astype(float))
	print('RMSE_5 of '  + ': ' + str(rmse_5))
	print('RMSE_25 of '  + ': ' + str(rmse_25))
	print('RMSE_70 of '  + ': ' + str(rmse_70))
	print('RMSE_150 of '  + ': ' + str(rmse_150))
	# RMSE = rmse(meanD['HRLDAS_0–10 cm'].to_numpy().astype(float),
	#             meanD['In situ_0–10 cm'].to_numpy().astype(float))
	# print(RMSE)
	# RMSE = rmse(meanD['HRLDAS_10–100 cm'].to_numpy().astype(float),
	#             meanD['In situ_10–100 cm'].to_numpy().astype(float))
	# print(RMSE)
	# RMSE = rmse(meanD.loc[pd.date_range(start="2020-06-15", end="2020-8-31"),'HRLDAS_0–10 cm'].to_numpy().astype(float),
	#             meanD.loc[pd.date_range(start="2020-06-15", end="2020-8-31"),'In situ_0–10 cm'].to_numpy().astype(float))
	# print(RMSE)
	# RMSE = rmse(meanD.loc[pd.date_range(start="2020-06-15", end="2020-8-31"),'HRLDAS_10–100 cm'].to_numpy().astype(float),
	#             meanD.loc[pd.date_range(start="2020-06-15", end="2020-8-31"),'In situ_10–100 cm'].to_numpy().astype(float))
	# print(RMSE)

	# model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE01_NONIRRI_2020040100.LDASOUT_DOMAIN1_ET.csv'
	# ground_measures=r'F:\work\modelCalibration\wsdata\NE01_2020.csv'
	# ne1_ET = AmerSitesET(model_output,ground_measures)
	#
	# model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE01_2020040100.LDASOUT_DOMAIN1_ET.csv'
	# ground_measures=r'F:\work\modelCalibration\wsdata\NE01_2020.csv'
	# ne1_ET_irri = AmerSitesET(model_output,ground_measures)
	#
	# model_output = r'C:\Users\nuds\Dropbox\singlepoint/data/2020042500_JOHNSON_IRRI.LDASOUT_DOMAIN1_ET.csv'
	# ground_measures = r'C:\Users\nuds\Dropbox\singlepoint/wsdata/2020_JOHNSON1_ET.csv'
	# ne3_ET = unlSitesET(model_output, ground_measures)
	#
	# ne3_ET = ne3_ET.iloc[1:, :]
	# # model_output = r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE02_2020040100.LDASOUT_DOMAIN1_ET.csv'
	# # ground_measures = r'F:\work\modelCalibration\wsdata\NE02_2020.csv'
	# # ne2_ET = AmerSitesET(model_output, ground_measures)
	#
	# mergeD = reduce(lambda a, b: a.add(b, fill_value=0), [ne1_ET, ne3_ET, ])
	# mergeD = mergeD.iloc[1:, :]
	# meanD = mergeD / 2
	#
	# mergeD_irri = reduce(lambda a, b: a.add(b, fill_value=0), [ne1_ET_irri, ne3_ET, ])
	# mergeD_irri = mergeD_irri.iloc[1:, :]
	# meanD_irri = mergeD_irri / 2
	#
	# meanD = pd.merge(meanD, meanD_irri, how='inner', right_index=True, left_index=True, suffixes=('', '_irri'))
	#
	# # meanD = ne1_ET.iloc[1:, :]
	# fig = plt.figure(figsize=(20, 5))
	# ax1 = fig.add_subplot(111)
	# meanD[['HRLDAS_ET_irri', 'In situ_ET']].plot(ax=ax1)
	# # meanD[['HRLDAS_ET', 'HRLDAS_ET_irri']].plot(ax=ax1)
	# plt.show()
	# RMSE = rmse(meanD['HRLDAS_ET'].to_numpy().astype(float),
	#             meanD['In situ_ET'].to_numpy().astype(float))
	# print(RMSE)
	# r2 =r2_score(meanD['HRLDAS_ET'].to_numpy().astype(float),
	#             meanD['In situ_ET'].to_numpy().astype(float))
	# print(r2)
	# scatter_plot(meanD['HRLDAS_ET'],
	#             meanD['In situ_ET'])

def scatter_plot(d1,d2,p_num=0, depth=0):
	######################
	# spearman r
	# global pearson r
	######################
	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111)
	ax1.plot(d1, d2, 'o')
	plt.ylabel(d2.name)
	plt.xlabel(d1.name)

	d1 = d1[:, np.newaxis]
	d2 = d2[:, np.newaxis]
	lrModel = LinearRegression()
	lrModel.fit(d1, d2)
	predicts = lrModel.predict(d1)
	R2 = lrModel.score(d1, d2)
	print('R2 = %.2f' % R2)
	coef = lrModel.coef_
	intercept = lrModel.intercept_

	plt.plot(d1,predicts, color = 'red', label='predicted value')
	ax1.set(title=f'y={coef[0][0]}*x+{intercept[0]} with R2={R2}')
	plt.legend()
	plt.show()

def valET():
	model_output =r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE01_2020040100.LDASOUT_DOMAIN1_ET.csv'
	ground_measures=r'F:\work\modelCalibration\wsdata\NE01_2020.csv'
	ne1_ET = AmerSitesET(model_output,ground_measures)

	fig = plt.figure(figsize=(20, 5))
	ax1 = fig.add_subplot(311)
	ne1_ET[['HRLDAS_ET', 'In situ_ET']].plot(ax=ax1)

	ne1_ET = ne1_ET.iloc[1:, :]
	RMSE = rmse(ne1_ET['HRLDAS_ET'].to_numpy().astype(float),
				ne1_ET['In situ_ET'].to_numpy().astype(float))
	print(RMSE)

	model_output = r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE02_2020040100.LDASOUT_DOMAIN1_ET.csv'
	ground_measures = r'F:\work\modelCalibration\wsdata\NE02_2020.csv'
	ne2_ET = AmerSitesET(model_output, ground_measures)

	ne2_ET = ne2_ET.iloc[1:, :]
	RMSE = rmse(ne2_ET['HRLDAS_ET'].to_numpy().astype(float),
				ne2_ET['In situ_ET'].to_numpy().astype(float))
	print(RMSE)

	ax2 = fig.add_subplot(312)
	ne2_ET[['HRLDAS_ET', 'In situ_ET']].plot(ax=ax2)


	model_output = r'C:\Users\nuds\Dropbox\ws_paper\major_revision\data\NE03_2020040100.LDASOUT_DOMAIN1_ET.csv'
	ground_measures = r'F:\work\modelCalibration\wsdata\NE03_2020.csv'
	ne3_ET = AmerSitesET(model_output, ground_measures)

	ne3_ET = ne3_ET.iloc[1:, :]
	RMSE = rmse(ne3_ET['HRLDAS_ET'].to_numpy().astype(float),
				ne3_ET['In situ_ET'].to_numpy().astype(float))
	print(RMSE)

	ax3 = fig.add_subplot(313)
	ne3_ET[['HRLDAS_ET', 'In situ_ET']].plot(ax=ax3)
	plt.show()

def valETunl():

	model_output = r'C:\Users\nuds\Dropbox\singlepoint/data/2020042500_JOHNSON_IRRI.LDASOUT_DOMAIN1_ET.csv'
	ground_measures =  r'C:\Users\nuds\Dropbox\singlepoint/wsdata/2020_JOHNSON1_ET.csv'
	ne3_ET = unlSitesET(model_output, ground_measures)

	ne3_ET = ne3_ET.iloc[1:, :]
	RMSE = rmse(ne3_ET['HRLDAS_ET'].to_numpy().astype(float),
				ne3_ET['In situ_ET'].to_numpy().astype(float))
	print(RMSE)
	fig = plt.figure(figsize=(20, 5))
	ax3 = fig.add_subplot(111)
	ne3_ET[['HRLDAS_ET', 'In situ_ET']].plot(ax=ax3)
	plt.show()

def valSM():
	sites =[]
	sites.append(["CRN_Harrison_2019", "01/01/2019", "2019123123", "CRNH0203-2019-NE_Harrison_20_SSE.txt"]);
	sites.append(["CRN_Harrison_2020", "01/01/2020", "2020123123", "CRNH0203-2020-NE_Harrison_20_SSE.txt"]);
	sites.append(["CRN_Harrison_2021", "01/01/2021", "2021123123", "CRNH0203-2021-NE_Harrison_20_SSE.txt"]);

	sites.append(["CRN_Lincoln_8ENE_2019",  "01/01/2019", "2019123123", "CRNH0203-2019-NE_Lincoln_8_ENE.txt"]);
	sites.append(["CRN_Lincoln_8ENE_2020",  "01/01/2020", "2020123123", "CRNH0203-2020-NE_Lincoln_8_ENE.txt"]);
	sites.append(["CRN_Lincoln_8ENE_2021",  "01/01/2021", "2021123123", "CRNH0203-2021-NE_Lincoln_8_ENE.txt"]);

	sites.append(["CRN_Lincoln_11SW_2019", "01/01/2019", "2019123123", "CRNH0203-2019-NE_Lincoln_11_SW.txt"]);
	sites.append(["CRN_Lincoln_11SW_2020", "01/01/2020", "2020123123", "CRNH0203-2020-NE_Lincoln_11_SW.txt"]);
	sites.append(["CRN_Lincoln_11SW_2021", "01/01/2021", "2021123123", "CRNH0203-2021-NE_Lincoln_11_SW.txt"]);

	sites.append(["CRN_Whitman_2019", "01/01/2019", "2019123123", "CRNH0203-2019-NE_Whitman_5_ENE.txt"]);
	sites.append(["CRN_Whitman_2020", "01/01/2020", "2020123123", "CRNH0203-2020-NE_Whitman_5_ENE.txt"]);
	sites.append(["CRN_Whitman_2021", "01/01/2021", "2021123123", "CRNH0203-2021-NE_Whitman_5_ENE.txt"]);

	sites.append(["SCAN_JohnsonFarm_2019", "01/01/2019", "2019123123", "JohnsonFarm.txt"]);
	sites.append(["SCAN_JohnsonFarm_2020", "01/01/2020", "2020123123", "JohnsonFarm.txt"]);
	sites.append(["SCAN_JohnsonFarm_2021", "01/01/2021", "2021123123", "JohnsonFarm.txt"]);

	sites.append(["SCAN_RogersFarm_2019", "01/01/2019", "2019123123", "RogerFarm.txt"]);
	sites.append(["SCAN_RogersFarm_2020",  "01/01/2020", "2020123123", "RogerFarm.txt"]);
	sites.append(["SCAN_RogersFarm_2021",  "01/01/2021", "2021123123", "RogerFarm.txt"]);

	sites.append(["SCAN_Torrington_2019", "01/01/2019", "2019123123", "Torrington.txt"]);
	sites.append(["SCAN_Torrington_2020", "01/01/2020", "2020123123", "Torrington.txt"]);
	sites.append(["SCAN_Torrington_2021", "01/01/2021", "2021123123", "Torrington.txt"]);

	sites.append(["SCAN_ShagbarkHills_2019",  "01/01/2019", "2019123123", "ShabarkHills.txt"]);
	sites.append(["SCAN_ShagbarkHills_2020",  "01/01/2020", "2020123123", "ShabarkHills.txt"]);
	sites.append(["SCAN_ShagbarkHills_2021",  "01/01/2021", "2021123123", "ShabarkHills.txt"]);

	sites.append(["AmeriFlux_NE01_2019", "01/01/2019","2019123123", "AMF_US-Ne1_BASE-BADM_12-5/AMF_US-Ne1_BASE_HR_12-5.csv"])
	sites.append(["AmeriFlux_NE01_2020", "01/01/2020","2020123123", "AMF_US-Ne1_BASE-BADM_12-5/AMF_US-Ne1_BASE_HR_12-5.csv"])
	sites.append(["AmeriFlux_NE01_2021", "01/01/2021","2021123123", "AMF_US-Ne1_BASE-BADM_12-5/AMF_US-Ne1_BASE_HR_12-5.csv"])

	sites.append(["AmeriFlux_NE02_2019",  "01/01/2019","2019123123", "AMF_US-Ne2_BASE-BADM_12-5/AMF_US-Ne2_BASE_HR_12-5.csv"])
	sites.append(["AmeriFlux_NE02_2020",  "01/01/2020","2020123123", "AMF_US-Ne2_BASE-BADM_12-5/AMF_US-Ne2_BASE_HR_12-5.csv"])
	sites.append(["AmeriFlux_NE02_2021",  "01/01/2021","2021123123", "AMF_US-Ne2_BASE-BADM_12-5/AMF_US-Ne2_BASE_HR_12-5.csv"])

	sites.append(["AmeriFlux_NE03_2019", "01/01/2019","2019123123", "AMF_US-Ne3_BASE-BADM_12-5/AMF_US-Ne3_BASE_HR_12-5.csv"])
	sites.append(["AmeriFlux_NE03_2020", "01/01/2020","2020123123", "AMF_US-Ne3_BASE-BADM_12-5/AMF_US-Ne3_BASE_HR_12-5.csv"])
	sites.append(["AmeriFlux_NE03_2021", "01/01/2021","2021123123", "AMF_US-Ne3_BASE-BADM_12-5/AMF_US-Ne3_BASE_HR_12-5.csv"])

	hourlyData = []
	dailyData = []

	for site in sites:
		start = dt.datetime.strptime(site[1],'%m/%d/%Y')
		end = dt.datetime.strptime(site[2],'%Y%m%d%H')
		startStr = dt.datetime.strftime(start, "%Y%m%d%H")
		endStr = site[2]
		start = dt.datetime(start.year, start.month+3, start.day)
		end = dt.datetime(end.year, end.month-2, end.day)
		ground_measures = r'E:\OneDrive - George Mason University - O365 Production\work/validation/site_data/'+site[3]
		model_data = r'E:\OneDrive - George Mason University - O365 Production\work/validation/model_data/'
		result_data = r'E:\OneDrive - George Mason University - O365 Production\work\validation\result\data/'
		arranged_output = result_data + site[0] + '.csv'
		if site[0].split('_')[0] == "CRN":
			model_ouput = model_data + site[0] + '_' + startStr + '.LDASOUT_DOMAIN1_SM.csv'
			siteSM = crnSM(model_ouput, ground_measures, start, end, arranged_output)
			if (siteSM.empty): continue
			hourlyData.append(printRMSE(siteSM, site))

		elif site[0].split('_')[0] == "SCAN":
			model_ouput = model_data + site[0] + '_' + startStr + '.LDASOUT_DOMAIN1_ET.csv'
			siteSM = scanSM(model_ouput, ground_measures, start, end, arranged_output)
			if (siteSM.empty): continue
			dailyData.append(printRMSE(siteSM, site))

		elif site[0].split('_')[0] == "AmeriFlux":
			model_ouput = model_data + site[0] + '_' + startStr + '.LDASOUT_DOMAIN1_SM.csv'
			siteSM = AmerSitesSM(model_ouput, ground_measures, start, end, arranged_output)
			if (siteSM.empty): continue
			hourlyData.append(printRMSE(siteSM, site))
	plotSM(hourlyData)
	plotSM(dailyData)
	plt.show()

def printRMSE(siteSM, site):
	rmse_list =[]
	bias_list=[]
	ubrmse_list=[]
	R2_list = []
	for j in ['5', '25', '70','150']:
		predictions = siteSM['HRLDAS_{} cm'.format(j)].to_numpy().astype(float)
		targets = siteSM['In situ_{} cm'.format(j)].to_numpy().astype(float)
		bias_values = bias(predictions,targets)
		rmse_list.append(rmse(predictions,targets))
		bias_list.append(bias_values)
		ubrmse_list.append(ubrmse(predictions, targets))
		R2_list.append(r2(predictions,targets))
		siteSM['HRLDAS_{} cm_unbiased'.format(j)] = predictions - bias_values
		print('RMSE_{} of {} {}: {}'.format(j,str(len(siteSM)), site[0], str(rmse_list[-1])))
		print('Bias_{} of {} {}: {}'.format(j, str(len(siteSM)), site[0], str(bias_list[-1])))
		print('ubRMSE_{} of {} {}: {}'.format(j, str(len(siteSM)), site[0], str(ubrmse_list[-1])))
		print('R2_{} of {} {}: {}'.format(j, str(len(siteSM)), site[0], str(R2_list[-1])))
	return({'year':int(site[2][:4]),'RMSE':rmse_list,'ubRMSE':ubrmse_list, 'Bias':bias_list,'R2':R2_list,
			'obser_num': len(siteSM), 'siteSM':siteSM, 'name':site[0]})


def plotSM(smData):
	for b in [ '_unbiased','']:
		plt.rc('font', size=12)
		fig, axs = plt.subplots(4,3)
		result_data = r'E:\OneDrive - George Mason University - O365 Production\work\validation\result\data/'
		for i in range(3):
			smData_year = list(filter(lambda a: a['year'] == 2019+i, smData))
			meanMetrics =[]
			for j in ['RMSE', 'Bias', 'ubRMSE', 'R2']:

				list_rmse = [[a[j],a['obser_num']] for a in smData_year]
				blist = [[i * a[1] for i in a[0]] for a in list_rmse]
				mergeRMSE = reduce(lambda a, b: list(map(add, a, b)), blist)
				meanMetrics.append([i / sum([i[1] for i in list_rmse]) for i in mergeRMSE])
			np.savetxt(result_data + str(2019+i)+'_'+str(len(smData)) + '_metrics.csv', meanMetrics,
					   delimiter=",", fmt='%s')
			smSite = [d['siteSM'] for d in smData_year]
			mergeD = reduce(lambda a, b: a.add(b, fill_value=0), smSite)
			mergeD = mergeD.dropna()
			meanD = mergeD / len(smData_year)
			meanD.loc[:, ['HRLDAS_5 cm{}'.format(b), 'In situ_5 cm']].plot(ax=axs[0,i])
			axs[0,i].set_title(" ubRMSE: %.4f" % meanMetrics[2][0])
			meanD.loc[:, ['HRLDAS_25 cm{}'.format(b), 'In situ_25 cm']].plot(ax=axs[1,i])
			axs[1, i].set_title("ubRMSE: %.4f" % meanMetrics[2][1])
			meanD.loc[:, ['HRLDAS_70 cm{}'.format(b), 'In situ_70 cm']].plot(ax=axs[2,i])
			axs[2, i].set_title("ubRMSE: %.4f" % meanMetrics[2][2])
			meanD.loc[:, ['HRLDAS_150 cm{}'.format(b), 'In situ_150 cm']].plot(ax=axs[3,i])
			axs[3, i].set_title("ubRMSE: %.4f" % meanMetrics[2][3])
		plt.gcf().set_size_inches(20, 10)
		plt.tight_layout()
		plt.savefig(r'E:\OneDrive - George Mason University - O365 Production\work\validation\result/'+str(len(smData))+b+'.png', dpi=100)
	#  plt.show()



def valSMAP():
	rmse_path = r'F:\work\validation\result\ee-chart.csv'
	rmse_hist = pd.read_csv(rmse_path)
	total_area = rmse_hist['diff_mean Count'].map(lambda x: float(str(x).replace(',',''))).dropna().sum()
	rmse_area = rmse_hist['diff_mean Count'].loc[rmse_hist['Band Value']<0.06].map(lambda x: float(str(x).replace(',',''))).dropna().sum()
	print(rmse_area/total_area)


if __name__ == '__main__':
	# main()
	#mergeData()
	#valET()
	#valETunl()
	start = dt.datetime.strptime('04/01/2021', '%m/%d/%Y')
	end = dt.datetime.strptime('2021103123', '%Y%m%d%H')
	valSM()
	#valSMAP()