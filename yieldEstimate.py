
import os
import os
os.environ['OPENBLAS_NUM_THREADS'] = '10'
import pandas as pd
import numpy as np
import datetime as dt
import netCDF4 as nc
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import subprocess
import time

from irriTable import readVariable, getTexture, getWP, getAWC
from AquaCropCalibration import getAquaCropSM

import re

# function to return the irrigation depth to apply on next day
def get_depth(model):
	t = model._clock_struct.time_step_counter # current timestep
	# get weather data for next 7 days
	weather10 = model._weather[t+1:min(t+10+1,len(model._weather))]
	# if it will rain in next 7 days
	if sum(weather10[:,2])>0:
		# check if soil is over 70% depleted
		if t>0 and model._init_cond.depletion/model._init_cond.taw > 0.7:
			depth=10
		else:
			depth=0
	else:
		# no rain for next 10 days
		depth=10
	return depth

def to_camel(s):
	return  re.sub(r"(_|-)+", " ", s).title().replace(" ","")

# def getTexture(lat, lon):
# 	texture_path = "./soilProps/"
# 	soil_texture = subprocess.check_output("gdallocationinfo -valonly -wgs84 "+texture_path+"texture_500m_remap.tif "+str(lon)+" "+str(lat), shell=True)
# 	texture_pair = pd.read_csv(texture_path+"texture_025.csv")
#
# 	texture_value = str(soil_texture).split("'")[1].split("\\")[0]
# 	texture_info = texture_pair.loc[texture_pair['value'] == int(texture_value)]
# 	texture_info.iloc[0,3] = to_camel(texture_info.iloc[0,3])
# 	return(texture_info)

def getSoilProp(lat, lon, prop):
	url = f'https://geobrain.csiss.gmu.edu/soil-properties/query_grid.php?lat={lat}&lon={lon}&property={prop}'
	html_text = requests.get(url).text
	soup = BeautifulSoup(html_text, 'html.parser')
	value = float(soup.select(".value")[0].get_text(strip=True))
	return value

# def getWP(lat, lon):
# 	texture_path = "./soilProps/"
# 	wilting_point = subprocess.check_output("gdallocationinfo -valonly -wgs84 "+texture_path+"wp_100cm_500m.tif "+str(lon)+" "+str(lat), shell=True)
# 	wp_value = float(wilting_point)
# 	return wp_value
#
# def getAWC(lat, lon):
# 	texture_path = "./soilProps/"
# 	awc = subprocess.check_output("gdallocationinfo -valonly -wgs84 "+texture_path+"awc_100cm_500m.tif "+str(lon)+" "+str(lat), shell=True)
# 	awc_value = float(awc)
# 	return awc_value

def getInitWC(ETPath):
	# initWC = []
	# layer_time = planting_date[:-2]+"16"
	# for i in range(4):
	# 	url = f"https://geobrain.csiss.gmu.edu/ncWMS2/wms?LAYERS=LDASOUT/M_{layer_time}.LDASOUT_DOMAIN1/SOIL_M" \
	# 		  f"&QUERY_LAYERS=LDASOUT/M_{layer_time}.LDASOUT_DOMAIN1/SOIL_M" \
	# 		  "&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetFeatureInfo" \
	# 		  f"&BBOX={lon-0.00005},{lat-0.00005},{lon+0.00005},{lat+0.00005}" \
	# 		  f"&FEATURE_COUNT=5&HEIGHT=600&WIDTH=750&FORMAT=image/png&INFO_FORMAT=text/xml&SRS=EPSG:4326&X=351&Y=420&ELEVATION={i}"
	# 	xml_text = requests.get(url).text
	# 	soup = BeautifulSoup(xml_text,'html.parser')
	# 	value =  float(soup.find("value").get_text(strip=True))
	# 	initWC.append(value)
	et= pd.read_csv(ETPath, index_col=0)
	et.index = et.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
	initWC = [et.SoilM1[0]*1, et.SoilM2[0]*1, et.SoilM3[0]*1, et.SoilM4[0]*1]
	return initWC

def trans_irr(irr_record):
	if len(irr_record) > 0:
		schedule = pd.DataFrame(irr_record)
		schedule = schedule[['start', 'volume']]
		schedule.columns = ['Date', 'Depth']
		schedule['Date'] = pd.to_datetime(schedule['Date']).dt.date
		return schedule
	else:
		return pd.DataFrame(columns=["Date", "Depth"])

def yield_est(lat, lon, awc, wp, planting_date, target_date, irr_record, x, y, site_name,dep_thld,ETorSM):
	#planting, target date: yyyyMMddHH
	result_path = r"/home/hzhao/single-point/newoutput/" + str(int(x)) + "/" + str(int(y)) + "/"

	soil_info = getTexture(lat,lon)
	# thwp = soil_info.iloc[0]['wp']
	# thfc = soil_info.iloc[0]['fc']
	ksat_mean = getSoilProp(lat, lon, 'mean_ksat') * 86.4

	thwp = getWP(lat, lon)/100
	thfc = thwp+getAWC(lat, lon)
	filepath = './input/dailyClimate.txt' # climate data for UNL sites
	# filepath = result_path+site_name+'_'+nc_file+'_dailyClimate.txt'
	weather_data = prepare_weather(filepath)
	yield_run(thwp, thfc, planting_date, target_date, irr_record, result_path, site_name, ETorSM, ksat_mean, soil_info, weather_data)

def yield_run(thwp, thfc, planting_date, target_date, irr_record, result_path, site_name, ETorSM, ksat_mean, soil_info, weather_data, write_to_file=True):
	nc_file = str(planting_date) + '.LDASOUT_DOMAIN1'
	end = dt.datetime.strptime(target_date, "%Y%m%d%H")
	endD = end - dt.timedelta(days=1)
	target_date = dt.datetime.strftime(endD, "%Y%m%d%H")
	####################################################
	# weather information
	####################################################
	# specify filepath to weather file

	# filepath = './input/dailyClimate.txt' # climate data for UNL sites
	# # filepath = result_path+site_name+'_'+nc_file+'_dailyClimate.txt'
	# weather_data = prepare_weather(filepath)

	####################################################
	# soil information
	####################################################
	custom_soil = Soil('custom',cn=soil_info.iloc[0]['cn'], rew=soil_info.iloc[0]['rew'],dz=[0.1]*10)
	custom_soil.add_layer(thickness=1.0,thWP=thwp,
					 thFC=thfc,thS=soil_info.iloc[0]['s'],Ksat=ksat_mean,
					 penetrability=100)
	####################################################
	# crop information
	####################################################
	start_date = str(planting_date)[4:6]+'/'+str(planting_date)[6:8]  # Planting Date (mm/dd)
	end_date = str(target_date)[4:6]+'/'+str(target_date)[6:8]
	year = str(planting_date)[:4]
	corn = Crop('Maize', planting_date=start_date,
				harvest_date=end_date,
				EmergenceCD=5,
				MaxRootingCD=68,
				SenescenceCD=109,
				MaturityCD=126,
				HIstartCD=74,
				YldFormCD=47,
				FloweringCD=13,
				Zmax=1.12,
				PlantPop=107692, # Numbers of plants per hectare
				CGC_CD=0.16312, # Canopy growth coefficient (CGC): Increase in canopy growth (in fraction per day)
				)

	# InitWC = InitWCClass(value=['FC'])
	####################################################
	# initial water content information
	####################################################

	# ETPath = result_path+site_name+'_'+nc_file+'_ET.csv'
	# InitWCValue = getInitWC(ETPath)
	# InitDepth = [0.1,0.3,0.6,1.0]
	# initWC = InitialWaterContent(wc_type='Num', method='Layer',
	# 					 depth_layer=InitDepth,value=InitWCValue)

	InitDepth = [0.25, 0.46, 0.6, 0.86]
	InitWC_dict = {"UNL_JOHNSON_2020": [0.20, 0.23, 0.28, 0.29], "UNL_KELLY_2020": [0.113, 0.109, 0.128, 0.139],
				   "UNL_HOME_2020": [0.142, 0.131, 0.139, 0.169],"UNL_NORTH_2020": [0.076, 0.063, 0.09, 0.10],
				   "UNL_HOME_2019": [0.086, 0.096, 0.135, 0.134],"UNL_EAST_2019": [0.08, 0.083, 0.115, 0.124],
				   "UNL_KELLY_2019": [0.12, 0.120, 0.122, 0.137], "UNL_LINKS_2019": [0.09, 0.11, 0.12, 0.14]
				   }
	InitWCValue = InitWC_dict[site_name]
	initWC = InitialWaterContent(wc_type='Num', method='Depth',
						 depth_layer=InitDepth,value=InitWCValue)
	# print(f'wp:{thwp} fc:{thfc} ksat:{ksat_mean} initWC:{InitWCValue}')
	####################################################
	# irrigation schedule information
	####################################################
	# irrigation schedule in Johnson 2020
	schedule_johnson2020 = pd.DataFrame({'Date': [f'5/1/{year}',f'6/12/{year}',f'6/18/{year}',
										  f'6/26/{year}',f'6/29/{year}',f'7/1/{year}',
										  f'7/5/{year}',f'7/10/{year}',f'7/17/{year}',f'7/23/{year}',
										  f'7/24/{year}',f'7/29/{year}',f'8/4/{year}',f'8/11/{year}',
										  f'8/19/{year}',f'8/22/{year}',f'8/28/{year}',f'9/2/{year}'],
								'Depth': [12.954,7.874,11.684,11.684,19.05,20.828,21.082,6.35,16.764,5.08,10.922,
											 20.32,25.146,21.844,21.59,21.59,21.59,21.59]})
	# irrigation schedule in Kelly 2020
	schedule_kelly2020 = pd.DataFrame({'Date': [f'5/1/{year}',f'6/12/{year}',f'6/18/{year}',
										  f'6/26/{year}',f'6/29/{year}',f'7/1/{year}',
										  f'7/5/{year}',f'7/10/{year}',f'7/17/{year}',f'7/24/{year}',
										  f'7/27/{year}',f'7/29/{year}',f'8/4/{year}',f'8/11/{year}',
										  f'8/20/{year}',f'8/28/{year}',f'9/2/{year}'],
								'Depth': [12.954,7.874,11.684,11.684,19.05,20.828,21.082,6.35,16.764,5.08,12.954,21.082,24.13,
											 22.098, 21.59,21.59,21.59]})
	schedule_home2020 = pd.DataFrame({'Date': [f'5/1/{year}',f'6/12/{year}',
										  f'6/26/{year}',f'6/29/{year}',f'7/1/{year}',
										  f'7/6/{year}',f'7/10/{year}',f'7/17/{year}',f'7/22/{year}',
										  f'7/24/{year}',f'7/29/{year}',f'8/4/{year}',f'8/11/{year}',
										  f'8/20/{year}',f'8/28/{year}',f'9/2/{year}'],
								'Depth': [13.716,7.874,13.716,22.098,23.114,23.114,6.604,16.764,5.08,11.43,22.86,25.146,
											 22.098, 21.59,21.59,21.59]})
	schedule_north2020 = pd.DataFrame({'Date': [f'5/1/{year}', f'6/12/{year}',f'6/18/{year}',
															  f'6/26/{year}', f'6/29/{year}', f'7/1/{year}',
											   f'7/5/{year}', f'7/10/{year}', f'7/17/{year}', f'7/22/{year}',
											   f'7/25/{year}', f'7/29/{year}', f'8/4/{year}', f'8/11/{year}',
											   f'8/20/{year}', f'8/28/{year}', f'9/2/{year}'],
									  'Depth': [12.7, 12.446, 14.224, 12.446, 20.066, 22.352, 22.352, 6.604, 16.51, 5.08,
												7.62, 20.574, 24.384,1.59, 21.59, 21.59, 21.59]})

	schedule_home2019 = pd.DataFrame({'Date': [ f'6/11/{year}', f'6/17/{year}',
												f'7/3/{year}',f'7/11/{year}', f'7/15/{year}', f'7/17/{year}',f'7/22/{year}',
												f'7/25/{year}', f'7/29/{year}', f'7/31/{year}', f'8/6/{year}',
												f'9/16/{year}'],
									   'Depth': [7.62,8.128,12.7,21.59,21.59,21.59,22.098,23.114,7.874,23.114,22.098,22.098]})
	schedule_east2019 = pd.DataFrame({'Date': [ f'6/11/{year}', f'6/17/{year}',
												f'7/2/{year}',f'7/11/{year}', f'7/15/{year}', f'7/17/{year}',f'7/22/{year}',
												f'7/25/{year}', f'7/29/{year}', f'7/31/{year}', f'8/6/{year}',
												f'9/16/{year}'],
									   'Depth': [8.89,9.398,19.812,21.59,21.59,21.59,21.59,21.59,6.858,8.636,20.066,20.066]})
	schedule_kelly2019 = pd.DataFrame({'Date': [f'6/11/{year}', f'6/21/{year}',
											   f'7/2/{year}', f'7/11/{year}', f'7/15/{year}', f'7/17/{year}',
											   f'7/22/{year}',
											   f'7/25/{year}', f'7/29/{year}', f'7/30/{year}',f'7/31/{year}', f'8/6/{year}',
											   f'9/16/{year}'],
									  'Depth': [10.16, 9.906, 19.558,15.24, 23.622, 23.622, 23.622, 23.622, 9.398, 7.366, 20.32,
												20.32, 20.32]})
	schedule_links2019 = pd.DataFrame({'Date': [f'6/11/{year}', f'6/21/{year}',
											   f'7/2/{year}', f'7/11/{year}', f'7/15/{year}', f'7/17/{year}',
											   f'7/22/{year}',
											   f'7/25/{year}', f'7/29/{year}', f'7/30/{year}',f'7/31/{year}', f'8/6/{year}',
											   f'9/16/{year}'],
									  'Depth': [10.16, 10.16, 20.32,14.732, 21.59, 21.59, 21.59, 21.59, 8.89, 6.858, 20.32,
												20.32, 20.32]})
	sche_dict = {"UNL_JOHNSON_2020":schedule_johnson2020, "UNL_KELLY_2020":schedule_kelly2020, "UNL_HOME_2020":schedule_home2020,
				 "UNL_NORTH_2020":schedule_north2020, "UNL_HOME_2019":schedule_home2019,"UNL_EAST_2019":schedule_east2019,
				 "UNL_KELLY_2019":schedule_kelly2019, "UNL_LINKS_2019":schedule_links2019
				 }

	schedule = sche_dict[site_name]

	schedule['Date'] = pd.to_datetime(schedule['Date'], infer_datetime_format=True)
	# print(schedule)

	#print(schedule)
	real_irr_schedule = IrrigationManagement(irrigation_method=3, Schedule=schedule, MaxIrr=60)
	recommend_schedule = trans_irr(irr_record)
	recommend_irr_schedule = IrrigationManagement(irrigation_method=3, Schedule=recommend_schedule, MaxIrr=60)
	rainfed = IrrigationManagement(irrigation_method=0,)
	# irrigate according to 4 different soil-moisture thresholds
	threshold4_irrigate = IrrigationManagement(irrigation_method=1,SMT=[40,60,70,30]*4)
	# irrigate every 7 days
	interval_7 = IrrigationManagement(irrigation_method=2,IrrInterval=7)
	net_irrigation = IrrigationManagement(irrigation_method=4,NetIrrSMT=70)

	# define labels to help after
	labels=[
		#'rainfed','four thresholds','interval',
		'real schedule',
		# 'net',
		'recommended schedule'+'_'+ETorSM]
	strategies = [
		#rainfed,threshold4_irrigate,interval_7,
		real_irr_schedule,
		#net_irrigation,
		recommend_irr_schedule
	]
	sim_start = f'{year}/{start_date}'
	sim_end = f'{year}/{end_date}'
	outputs=[]
	irr_outputs=[]
	cropg_outputs=[]
	for i,irr_mngt in enumerate(strategies): # for both irrigation strategies...
		corn.Name = labels[i] # add helpfull label
		model = AquaCropModel(sim_start,
							sim_end,
							weather_data,
							custom_soil,
							corn,
							initial_water_content=initWC,
							irrigation_management=irr_mngt) # create model
		model.run_model(till_termination=True) # run model till the end
		outputs.append(model._outputs.final_stats) # save results
		irr_outputs.append(model._outputs.water_flux)
		cropg_outputs.append(model._outputs.crop_growth)

	# create model with IrrMethod= Constant depth
	# corn.Name = 'weather'  # add helpfull label
	# model = AquaCropModel(sim_start, sim_end, weather_data, custom_soil, corn, initial_water_content=initWC,
	# 					  irrigation_management=IrrigationManagement(irrigation_method=5, ))
	# model._initialize()
	# while model._clock_struct.model_is_finished is False:
	# 	# get depth to apply
	# 	depth = get_depth(model)
	# 	model._param_struct.IrrMngt.depth = depth
	# 	model.run_model(initialize_model=False)
	# outputs.append(model._outputs.final_stats) # save results
	# irr_outputs.append(model._outputs.water_flux)
	# labels.append('weather')

	dflist=outputs
	outlist=[]
	for i in range(len(dflist)):
		temp = pd.DataFrame(dflist[i][['Yield (tonne/ha)','Seasonal irrigation (mm)']])
		temp['label']=labels[i]
		outlist.append(temp)
		if write_to_file:
			irr_outputs[i].to_csv(result_path + site_name + '_'+labels[i] + '_daily_water_flux.csv')
			cropg_outputs[i].to_csv(result_path + site_name + '_' + labels[i] + '_crop_growth.csv')
	results = pd.concat(outlist,axis=0)

	#print(results)
	#results.to_csv('./output/yield_estimates.csv')
	if write_to_file:
		print(results)
		results.to_csv(result_path + site_name  +'_'+ETorSM+ '_yield_estimates.csv')
		recommend_schedule.to_csv(result_path + site_name  +'_'+ETorSM+ '_recommend_irr_schedule.csv')
	return results

def plot_result(results):
	# create figure consisting of 2 plots
	fig,ax=plt.subplots(2,1,figsize=(10,14))

	# create two box plots
	sns.barplot(data=results,x='label',y='Yield (tonne/ha)',ax=ax[0])
	sns.barplot(data=results,x='label',y='Seasonal irrigation (mm)',ax=ax[1])

	# labels and font sizes
	ax[0].tick_params(labelsize=15)
	ax[0].set_xlabel(' ')
	ax[0].set_ylabel('Yield (t/ha)',fontsize=18)

	ax[1].tick_params(labelsize=15)
	ax[1].set_xlabel(' ')
	ax[1].set_ylabel('Total Irrigation (ha-mm)',fontsize=18)

	plt.legend(fontsize=18)
	plt.show()

def run_hrldas_model(lat, lon, awc, wp, planting_date, target_date, irr_record, x, y, site_name,dep_thld,ETorSM): # update sm and et in the days following irrigation
	# start_string = start.strftime("%Y%m%d%H")
	start = dt.datetime.strptime(planting_date, "%Y%m%d%H")
	end = dt.datetime.strptime(target_date, "%Y%m%d%H")
	filePath = r"/home/hzhao/single-point/newoutput/" + str(int(x)) + "/" + str(int(y)) + "/"
	irrigation_data = "\'[]\'"
	nc_file = planting_date + '.LDASOUT_DOMAIN1'
	# cmd = "./run_hrldas.sh " + str(x) + " " + str(y) + " " + planting_date\
	# 	  + " " + str(target_date) + " " + str(irrigation_data) + " " + str(
	# 	lon) + " " + str(lat)
	# print(cmd)
	# time_start = time.perf_counter()
	# p = subprocess.Popen(cmd, shell=True)
	# p.wait()
	# time_end = time.perf_counter()
	# print('time cost: ', time_end - time_start, 's')
	sm = readVariable(nc_file, start, end,filePath,site_name)

if __name__ == '__main__':	# lat = 41.9406
	# lon = -98.1837
	# prop = 'mean_ksat'
	# planting_date = '2020042500'
	# target_date = '2020091800'
	# irr_record = \
	# 	[{"start": "2020-04-25T00:00:00.000Z", "end": "2020-04-25T00:00:00.000Z", "volume": 40.48956, "changed": True},
	# 	 {"start": "2020-06-30T00:00:00.000Z", "end": "2020-06-30T00:00:00.000Z", "volume": 39.31837, "changed": True},
	# 	 {"start": "2020-07-27T00:00:00.000Z", "end": "2020-07-27T00:00:00.000Z", "volume": 39.00198, "changed": True},
	# 	 {"start": "2020-08-05T00:00:00.000Z", "end": "2020-08-05T00:00:00.000Z", "volume": 38.86181, "changed": True},
	# 	 {"start": "2020-08-20T00:00:00.000Z", "end": "2020-08-20T00:00:00.000Z", "volume": 41.66939, "changed": True},
	# 	 {"start": "2020-09-03T00:00:00.000Z", "end": "2020-09-03T00:00:00.000Z", "volume": 39.52491, "changed": True}, ]
	# 	# [{"start": "2020-05-01T00:00:00.000Z", "end": "2020-05-01T00:00:00.000Z", "volume": 12.954, "changed": True},
	# 	#  {"start": "2020-06-12T00:00:00.000Z", "end": "2020-06-12T00:00:00.000Z", "volume": 7.874, "changed": True},
	# 	#  {"start": "2020-06-18T00:00:00.000Z", "end": "2020-06-18T00:00:00.000Z", "volume": 11.684, "changed": True},
	# 	#  {"start": "2020-06-26T00:00:00.000Z", "end": "2020-06-26T00:00:00.000Z", "volume": 11.684, "changed": True},
	# 	#  {"start": "2020-06-29T00:00:00.000Z", "end": "2020-06-29T00:00:00.000Z", "volume": 19.05, "changed": True},
	# 	#  {"start": "2020-07-01T00:00:00.000Z", "end": "2020-07-01T00:00:00.000Z", "volume": 20.828, "changed": True},
	# 	#  {"start": "2020-07-05T00:00:00.000Z", "end": "2020-07-05T00:00:00.000Z", "volume": 21.082, "changed": True},
	# 	#  {"start": "2020-07-10T00:00:00.000Z", "end": "2020-07-10T00:00:00.000Z", "volume": 6.35, "changed": True},
	# 	#  {"start": "2020-07-17T00:00:00.000Z", "end": "2020-07-17T00:00:00.000Z", "volume": 16.764, "changed": True},
	# 	#  {"start": "2020-07-23T00:00:00.000Z", "end": "2020-07-23T00:00:00.000Z", "volume": 5.08, "changed": True},
	# 	#  {"start": "2020-07-24T00:00:00.000Z", "end": "2020-07-24T00:00:00.000Z", "volume": 10.922, "changed": True},
	# 	#  {"start": "2020-07-29T00:00:00.000Z", "end": "2020-07-29T00:00:00.000Z", "volume": 20.32, "changed": True},
	# 	#  {"start": "2020-08-04T00:00:00.000Z", "end": "2020-08-04T00:00:00.000Z", "volume": 25.146, "changed": True},
	# 	#  {"start": "2020-08-11T00:00:00.000Z", "end": "2020-08-11T00:00:00.000Z", "volume": 21.844, "changed": True},
	# 	#  {"start": "2020-08-19T00:00:00.000Z", "end": "2020-08-19T00:00:00.000Z", "volume": 21.59, "changed": True},
	# 	#  {"start": "2020-08-22T00:00:00.000Z", "end": "2020-08-22T00:00:00.000Z", "volume": 21.59, "changed": True},
	# 	#  {"start": "2020-08-28T00:00:00.000Z", "end": "2020-08-28T00:00:00.000Z", "volume": 21.59, "changed": True},
	# 	#  {"start": "2020-09-02T00:00:00.000Z", "end": "2020-09-02T00:00:00.000Z", "volume": 21.59, "changed": True}, ]
	# awc = 172.2
	# wp = 282.052
	# x = 984
	# y = 466
	# site_name = 'UNL_JOHNSON_2020'
	#main()

	para_kelly2020 = {'lon':-98.2034475, 'lat':41.9480136,'wp':0,'awc':0,'planting_date':'2020050100', 'target_date':'2020103123', 'irr_record':[],
					  'x':986, 'y':448, 'site_name':"UNL_KELLY_2020", 'dep_thld':[50,50,50,50],'ETorSM':"TEST"}
	para_links2019 = {'lon':-98.21542, 'lat':41.95572,'wp':0,'awc':0,'planting_date':'2019042400', 'target_date':'2019103123', 'irr_record':[],
					  'x':984, 'y':450, 'site_name':"UNL_LINKS_2019", 'dep_thld':[50,50,50,50],'ETorSM':"TEST"}
	para_kelly2019 = {'lon':-98.2034475, 'lat':41.9480136,'wp':0,'awc':0,'planting_date':'2019042500', 'target_date':'2019103123', 'irr_record':[],
					  'x':986, 'y':448, 'site_name':"UNL_KELLY_2019", 'dep_thld':[50,50,50,50],'ETorSM':"TEST"}
	para_home2019 = {'lon':-98.1970, 'lat':41.94144,'wp':0,'awc':0,'planting_date':'2019050400', 'target_date':'2019103123', 'irr_record':[],
					  'x':987, 'y':447, 'site_name':"UNL_HOME_2019", 'dep_thld':[50,50,50,50],'ETorSM':"TEST"}
	para_home2020 = {'lon':-98.1970, 'lat':41.94144,'wp':0,'awc':0,'planting_date':'2020050100', 'target_date':'2020103123', 'irr_record':[],
					  'x':987, 'y':447, 'site_name':"UNL_HOME_2020", 'dep_thld':[50,50,50,50],'ETorSM':"TEST"}
	para_east2019 = {'lon':-98.18372, 'lat':41.9406,'wp':0,'awc':0,'planting_date':'2019050200', 'target_date':'2019103123', 'irr_record':[],
					  'x':989, 'y':447, 'site_name':"UNL_EAST_2019", 'dep_thld':[50,50,50,50],'ETorSM':"TEST"}
	para_johnson2020 = {'lon':-98.21608,'lat':42.02841,'wp':0,'awc':0, 'planting_date':'2020042500', 'target_date':'2020103123', 'irr_record':[],
					 'x': 984, 'y': 466, 'site_name': "UNL_JOHNSON_2020", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "TEST"}
	para_north2020 = {'lon':-98.19706, 'lat':41.94675, 'wp':0, 'awc':0, 'planting_date':'2020050800', 'target_date':'2020103123', 'irr_record':[],
					 'x': 987, 'y': 448, 'site_name': "UNL_NORTH_2020", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "TEST"}

	para_dict = {"UNL_JOHNSON_2020":para_johnson2020, "UNL_KELLY_2020":para_kelly2020, "UNL_HOME_2020":para_home2020,
				 "UNL_NORTH_2020":para_north2020, "UNL_HOME_2019":para_home2019,"UNL_EAST_2019":para_east2019,
				 "UNL_KELLY_2019":para_kelly2019, "UNL_LINKS_2019":para_links2019
				 }
	for key, value in para_dict.items():
		print(f"######################################{key}")
		# run_hrldas_model(**value)
		yield_est(**value)
	# getAquaCropSM()
	# yield_est(**para_kelly2020)