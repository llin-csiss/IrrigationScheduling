import pandas as pd
import numpy as np
import datetime as dt
import netCDF4 as nc
import logging
import subprocess
import re
from osgeo import gdal, ogr, osr
import affine
import math


logging.basicConfig(filename='/home/hzhao/sample/irrTable.log', level=logging.DEBUG,
					format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)


def to_camel(s):
	return  re.sub(r"(_|-)+", " ", s).title().replace(" ","")


def transform_coordinate(lat, lon):
	InSR = osr.SpatialReference()
	InSR.ImportFromEPSG(4326)  # WGS84/Geographic
	OutSR = osr.SpatialReference()
	OutSR.ImportFromEPSG(5072)  #

	Point = ogr.Geometry(ogr.wkbPoint)
	Point.AddPoint(lat, lon)  # use your coordinates here
	Point.AssignSpatialReference(InSR)  # tell the point what coordinates it's in
	Point.TransformTo(OutSR)  # project it to the out spatial reference
	# print('{0},{1}'.format(Point.GetX(), Point.GetY()))  # output projected X and Y coordinates
	return Point.GetX(), Point.GetY()

def get_raster_value(geo_x, geo_y, ds, band_index=1):
	"""Return raster value that corresponds to given coordinates."""
	# forward_transform = ds.GetGeoTransform()
	# reverse_transform = gdal.InvGeoTransform(forward_transform)
	# pixel_coord = gdal.ApplyGeoTransform(reverse_transform, geo_x, geo_y)
	# pixel_x = math.floor(pixel_coord[0])
	# pixel_y = math.floor(pixel_coord[1])
	forward_transform = \
		affine.Affine.from_gdal(*ds.GetGeoTransform())
	reverse_transform = ~forward_transform
	px, py = reverse_transform * (geo_x, geo_y)
	# print(px,py)
	px, py = int(px), int(py)
	pixel_coord = px, py
	# print(pixel_coord)
	pixel_x = math.floor(pixel_coord[0])
	pixel_y = math.floor(pixel_coord[1])
	band = ds.GetRasterBand(band_index)
	val_arr = band.ReadAsArray(pixel_x, pixel_y, 1, 1) # Avoid reading the whole raster into memory - read 1x1 array
	return val_arr[0][0]

def getVariablePd(nc_data, variable_name, columns):
	var_array = nc_data.variables[variable_name][:]
	hour_num = len(var_array)
	z = len(columns)
	var_array = np.reshape(var_array,(hour_num,z))
	var_pd = pd.DataFrame(var_array, columns=columns)
	return var_pd

def getVariablePdDaily(nc_data, variable_name, columns, accumulated=False, is_hourly=False, hour=23, maxV=False, minV=False):
	var_array = nc_data.variables[variable_name][:]
	hour_num = len(var_array)
	day_num = int((hour_num - 1) / 24)
	z = len(columns)
	second = 1 if is_hourly else 3600
	var_array = np.reshape(var_array, (hour_num, z))
	var_daily = np.reshape(var_array[:-1],(day_num, 24,z))
	if maxV:
		var_daily_max = [max(i) for i in var_daily]
		return pd.DataFrame(var_daily_max, columns=columns)
	if minV:
		var_daily_min = [min(i) for i in var_daily]
		return pd.DataFrame(var_daily_min, columns=columns)
	var_daily = [i[hour] if accumulated else sum((abs(a)+a)/2 for a in i)*second for i in var_daily]
	var_daily_pd = pd.DataFrame(var_daily, columns=columns)

	return var_daily_pd


def getWP(lat, lon):
	texture_path = "./soilProps/"
	filename = texture_path+"wp_100cm_500m.tif"
	dataset = gdal.Open(filename)
	geo_x, geo_y = transform_coordinate(lat, lon)
	wp = get_raster_value(geo_x,geo_y,dataset)
	# print(wp)
	# wilting_point = subprocess.check_output("gdallocationinfo -valonly -wgs84 "+texture_path+"wp_100cm_500m.tif "+str(lon)+" "+str(lat), shell=True)
	# wp_value = float(wilting_point)
	# print(wp_value)
	return(wp)

def getAWC(lat, lon):
	texture_path = "./soilProps/"
	filename = texture_path+"awc_100cm_500m.tif"
	dataset = gdal.Open(filename)
	geo_x, geo_y = transform_coordinate(lat, lon)
	awc = get_raster_value(geo_x, geo_y, dataset)
	# #subprocess requests too much memory
	# awc = subprocess.check_output("gdallocationinfo -valonly -wgs84 "+texture_path+"awc_100cm_500m.tif "+str(lon)+" "+str(lat), shell=True)
	# awc_value = float(awc)
	return awc

def getTexture(lat, lon):
	texture_path = "./soilProps/"
	filename = texture_path+"texture_500m_remap.tif"
	dataset = gdal.Open(filename)
	geo_x, geo_y = transform_coordinate(lat, lon)
	texture_value = get_raster_value(geo_x, geo_y, dataset)
	# print(texture_value)
	# soil_texture = subprocess.check_output("gdallocationinfo -valonly -wgs84 "+texture_path+"texture_500m_remap.tif "+str(lon)+" "+str(lat), shell=True)
	# print(soil_texture)# b'2\n'
	# texture_value = str(soil_texture).split("'")[1].split("\\")[0]
	texture_pair = pd.read_csv(texture_path+"texture_025.csv")
	texture_info = texture_pair.loc[texture_pair['value'] == int(texture_value)]
	texture_info.iloc[0,3] = to_camel(texture_info.iloc[0,3])
	return(texture_info)

def readVariable(nc_file, start, end, result_path, site_name, write_to_file=True):
	nc_path = result_path
	#nc_file = '2020033118.LDASOUT_DOMAIN1'
	nc_data = nc.Dataset(nc_path+nc_file, 'r+')

	ecan = getVariablePd(nc_data,'ECAN',['ET'])
	etran = getVariablePd(nc_data, 'ETRAN', ['ET'])
	edir = getVariablePd(nc_data, 'EDIR', ['ET'])
	lai = getVariablePd(nc_data, 'LAI', ['LAI'])
	fveg = getVariablePd(nc_data, 'FVEG', ['FVEG'])
	rainrate = getVariablePd(nc_data, 'RAINRATE', ['RAINRATE'])
	soilM = getVariablePd(nc_data, 'SOIL_M', [ 'SoilM1', 'SoilM2', 'SoilM3','SoilM4'])
	ugdrnoff = getVariablePd(nc_data, 'UGDRNOFF',['UGDRNOFF'])
	sfcrnoff = getVariablePd(nc_data,'SFCRNOFF',['SFCRNOFF'])
	gdd = getVariablePd(nc_data, 'GDD', ['GDD'])
	grain =getVariablePd(nc_data, 'GRAIN', ['GRAIN'])
	lfmass = getVariablePd(nc_data, 'LFMASS',['LFMASS'])
	rtmass = getVariablePd(nc_data, 'RTMASS', ['RTMASS'])
	stmass = getVariablePd(nc_data, 'STMASS', ['STMASS'])
	wood = getVariablePd(nc_data, 'WOOD', ['WOOD'])

	ecanD = getVariablePdDaily(nc_data,'ECAN', ['ET'])
	etranD = getVariablePdDaily(nc_data, 'ETRAN', ['ET'])
	edirD = getVariablePdDaily(nc_data,'EDIR', ['ET'])
	rainrateD = getVariablePdDaily(nc_data, 'RAINRATE', ['RAIN'], is_hourly=True)
	ugdrnoffD = getVariablePdDaily(nc_data, 'UGDRNOFF',['UGDRNOFF'], accumulated=True)
	sfcrnoffD = getVariablePdDaily(nc_data, 'SFCRNOFF', ['RUNOFF'], accumulated=True)
	smD = getVariablePdDaily(nc_data,'SOIL_M', [ 'SoilM1', 'SoilM2', 'SoilM3','SoilM4'],accumulated=True,hour=16)
	maxT = getVariablePdDaily(nc_data,'T2MV',['Tmax(K)'],maxV=True)
	minT = getVariablePdDaily(nc_data,'T2MV',['Tmin(K)'],minV=True)
	maxT['Tmax(C)'] = maxT['Tmax(K)']-273.15
	minT['Tmin(C)'] = minT['Tmin(K)']-273.15


	daily_irri = [0]*len(ecanD)
	irriD = pd.DataFrame(daily_irri, columns=['IRRI'])

	et = (ecan.abs() + ecan + etran.abs() + etran + edir.abs() + edir)/2
	etD = ecanD+etranD+edirD
	etD.at[0,'ET']=0

	startD = start #dt.datetime(2020, 4, 1)
	endD = end - dt.timedelta(days=1)#dt.datetime(2020, 12, 30)
	indexD = pd.date_range(startD, endD, freq='D')

	climateDaily = pd.merge(etD, rainrateD, how='inner', right_index=True, left_index=True)
	climateDaily = pd.merge(climateDaily, maxT['Tmax(C)'], how='inner', right_index=True, left_index=True)
	climateDaily = pd.merge(climateDaily, minT['Tmin(C)'], how='inner', right_index=True, left_index=True)
	climateDaily['Year'] = indexD.year
	climateDaily['Month'] = indexD.month
	climateDaily['Day'] = indexD.day
	climateDaily = climateDaily[['Day','Month','Year','Tmin(C)','Tmax(C)','RAIN','ET']]
	climateDaily.columns = ['Day','Month','Year','Tmin(C)','Tmax(C)','Prcp(mm)','Et0(mm)']
	if write_to_file:
		climateDaily.to_csv(result_path+site_name+'_'+nc_file+'_dailyClimate.txt', index=None, sep=' ')

	etD.index = indexD
	rainrateD.index = indexD
	irriD.index = indexD
	ugdrnoffD.index = indexD
	sfcrnoffD.index = indexD
	smD.index = indexD
	etD = pd.merge(etD, rainrateD, how='inner', right_index=True, left_index=True)
	etD = pd.merge(etD, irriD, how='inner', right_index=True, left_index=True)
	etD = pd.merge(etD, sfcrnoffD, how='inner', right_index=True, left_index=True)
	etD = pd.merge(etD, ugdrnoffD, how='inner', right_index=True, left_index=True)
	etD = pd.merge(etD, smD, how='inner', right_index=True, left_index=True)
	etD['RDepth'] = [i * 7 for i in range(40)] + [i * 30 + 7*40 for i in range(30)]+[1120]*(len(etD)-70)
	etD.loc[etD.RDepth > 1120, 'RDepth'] = 1120
	sm_array = []
	for i in range(len(etD)):
		if i*7<=100:
			sm_array.append(etD.SoilM1[i])
		elif i*7>100 and i<=40:
			sm_array.append((etD.SoilM1[i]*100+etD.SoilM2[i]*(i*7-100))/(i*7))
		elif i > 40 and (i-40)*30 + 280 <= 400:
			sm_array.append((etD.SoilM1[i] * 100 + etD.SoilM2[i] * (180+(i-40)*30)) / (280+(i-40)*30))
		elif (i-40)*30+280>400 and (i-40)*30+280<=1000:
			sm_array.append((etD.SoilM1[i] * 100 + etD.SoilM2[i] * 300 + etD.SoilM3[i]*((i-40)*30-120)) / ((i-40)*30+280))
		elif (i-40)*30+280>1000 and (i-40)*30+280<=1120:
			sm_array.append(
				(etD.SoilM1[i] * 100 + etD.SoilM2[i] * 300 + etD.SoilM3[i] * 600 + etD.SoilM4[i] * ((i-40)*30-720)) / ((i-40)*30+280))
		else:
			sm_array.append(
				(etD.SoilM1[i] * 100 + etD.SoilM2[i] * 300 + etD.SoilM3[i] * 600 + etD.SoilM4[i] * 120) / 1120)
	etD['SM'] = sm_array
	#print(etD)
	#file_name = str(round(1*dep_thld[0]))
	if write_to_file:
		etD.to_csv(result_path+site_name+'_'+nc_file+'_ET.csv')

	# start = dt.datetime(2020, 3, 31, 18)
	# end = dt.datetime(2020, 12, 30, 18)
	endH = end.replace(hour=0)  # dt.datetime(2020, 12, 30)
	index = pd.date_range(start, endH, freq='H')
	#print(index)
	soilM.index = index
	et.index = index
	lai.index = index
	fveg.index = index
	rainrate.index = index
	ugdrnoff.index = index
	sfcrnoff.index = index
	gdd.index = index
	grain.index = index
	lfmass.index = index
	rtmass.index = index
	stmass.index = index
	wood.index = index
	#print(et)
	soilM =  pd.merge(soilM, et, how='inner', right_index=True, left_index=True)
	soilM =  pd.merge(soilM, lai, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, rainrate, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, ugdrnoff, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, sfcrnoff, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, gdd, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, grain, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, lfmass, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, rtmass, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, stmass, how='inner', right_index=True, left_index=True)
	soilM = pd.merge(soilM, wood, how='inner', right_index=True, left_index=True)
	#print(soilM)
	if write_to_file:
		soilM.to_csv(result_path+site_name+'_'+nc_file+'_SM.csv')

	nc_data.close()
	return etD


def buildTable(start, end, nc_file, crop_type, irri_record, result_path, awc, wp, site_name,dep_thld, ETorSM):
	#file_name = str(round(1*dep_thld[0]))
	ETPath = result_path+site_name+'_'+nc_file+ '_ET.csv'
	et= pd.read_csv(ETPath, index_col=0)
	et.index = et.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
	kc = buildKC(start, end, crop_type)
	et.iloc[0,0]=0
	et = et.astype({"ET": float})
	et["awc"] = awc
	et["wp"] = wp
	# et['RDepth'] = [i * 7 for i in range(40)] + [i * 30 + 7*40 for i in range(30)]+[1120]*(len(et)-70)
	# et.loc[et.RDepth > 1120, 'RDepth'] = 1120
	et['WUa'] = et['ET']*kc[crop_type]
	et["WUac"] = et['WUa'].cumsum()
	et["WUac"] = et['WUac'] + et['UGDRNOFF']+et['RUNOFF']
	et['Rc'] = et['RAIN'].cumsum()
	initIrriDate = -2
	if len(irri_record)>0:
		for item in irri_record:
			et.loc[ item["start"].split('T')[0].replace('-',''), 'IRRI'] = \
				et.loc[ item["start"].split('T')[0].replace('-',''), 'IRRI'] + item['volume']
		initIrriDate = np.max(np.nonzero(et.IRRI.tolist()))
	#et['Ic'] = et['IRRI'].cumsum()
	et['TotW'] = et['Rc']#+et['Ic']
	et['WIefc'] = et['TotW']
	#et['Dep'] = initDep
	# always start from the planting date
	initWC = et.SoilM1[0]*0.1 + et.SoilM2[0]*0.3 + et.SoilM3[0]*0.6  # initial depletion only considers soil moisture in top 10 cm
	initDep=awc+wp - initWC if awc+wp>initWC else 0# depletion ratio
	for i in range(len(et['WUac'])):
		if et['WUac'].iloc[i] + initDep*et['RDepth'].iloc[i] < et['WIefc'].iloc[i]:# when efficient water input is larger than depletion,
															# the overflow part should be subtracted
			temp = et['WIefc'].iloc[i:] - (et['WIefc'].iloc[i] - et['WUac'].iloc[i]-initDep * et['RDepth'].iloc[i])   #et.loc[start+dt.timedelta(days=day_num if day_num<em[crop_type] else em[crop_type]), 'TotW']
			et['WIefc'].iloc[i:] = temp
	et.loc[et['WIefc']<0, 'WIefc'] = 0
	et['Dep'] = et['WUac'] - et['WIefc']+initDep*et['RDepth']
	et.loc[et.Dep<0, 'Dep'] = 0

	et['IRRIre'] =0
	# thld_dict = {"UNL_JOHNSON_2020":0.5, "UNL_KELLY_2020":0.35, "UNL_HOME_2020":0.5,
	# 			 "UNL_NORTH_2020":0.5, "UNL_HOME_2019":0.5,"UNL_EAST_2019":0.5,
	# 			 "UNL_KELLY_2019":0.5, "UNL_LINKS_2019":0.5
	# 			 }#threshold triggering irrigation
	thld_dict = {"UNL_JOHNSON_2020": 0.8, "UNL_KELLY_2020": 0.35, "UNL_HOME_2020": 0.5,
				 "UNL_NORTH_2020": 0.5, "UNL_HOME_2019": 0.5, "UNL_EAST_2019": 0.5,
				 "UNL_KELLY_2019": 0.5, "UNL_LINKS_2019": 0.5
				 }  # threshold triggering irrigation
	#dep_thld = thld_dict[site_name]
	dep_thld_list = [dep_thld[0]]*25+ [dep_thld[1]]*35+ [dep_thld[2]]*40+ [dep_thld[3]]*500
	#four thresholds corresponding to four stages
	for i in range(len(et['Dep'])):
		if (et['Dep'].iloc[i] > dep_thld_list[i]*awc*et['RDepth'].iloc[i]/100) and (i > initIrriDate):
			# do not irrigate on the planting date
			et['IRRIre'].iloc[i] = (et['Dep'].iloc[i] -awc*0.1*et['RDepth'].iloc[i])\
									- np.multiply(et['RAIN'][i+1:i+6].values, [0.9,0.9**2, 0.9**3,0.9**4,0.9**5]).sum()
			#minus forecasted rainfall
			# #keep water depletion around 10% of fc
			if (et['IRRIre'].iloc[i] < 5) or (i-initIrriDate < 3):
				# do not irrigate within 3 days after last irrigation
				# do not irrigate if suggested amount less than 5 mm
				et['IRRIre'].iloc[i] = 0
				continue
			if et['IRRIre'].iloc[i] > 25:
				et['IRRIre'].iloc[i] = 25
			initIrriDate = i
			temp = et['TotW'].iloc[i:] + et['IRRIre'].iloc[i]
			et['TotW'].iloc[i:] = temp
			temp = et['WIefc'].iloc[i:] + et['IRRIre'].iloc[i]
			et['WIefc'].iloc[i:] = temp
			for j in range(i,len(et['WIefc'])):
				if et['WUac'].iloc[j]+initDep*et['RDepth'].iloc[i] < et['WIefc'].iloc[j]:
					temp = et['WIefc'].iloc[j:] - (et['WIefc'].iloc[j] - et['WUac'].iloc[j]-initDep*et['RDepth'].iloc[i])
					et['WIefc'].iloc[j:] = temp
			et.loc[et['WIefc'] < 0, 'WIefc'] = 0
			et['Dep'] = et['WUac'] - et['WIefc']+initDep*et['RDepth']
			et.loc[et.Dep < 0, 'Dep'] = 0
			break# only calculate the first irrigating amount

	et.to_csv(result_path + site_name+'_' + nc_file +'_'+ETorSM+ '_IRRI_TABLE.csv')
	recom = et.loc[et.IRRIre>0, 'IRRIre']
	if len(recom)>0:
		print("Recommended irrigation volume: "+str(recom[0])+" mm on date: "+str(recom.index[0]) )
		return recom
	else:
		return 0

def buildTableSM(nc_file, irri_record, result_path, awc, wp, site_name,dep_thld,ETorSM):
	# awc = awc/2
	# wp = wp/2
	#dep_thld = 0.5  # threshold triggering irrigation
	#file_name = str(round(1 * dep_thld[0]))
	ETPath = result_path + site_name+'_' + nc_file + '_ET.csv'
	et= pd.read_csv(ETPath, index_col=0)
	et.index = et.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
	et.iloc[0,0]=0
	et = et.astype({"ET": float})
	# et['RDepth'] = [i * 7 for i in range(40)] + [i * 30 + 7*40 for i in range(30)]+[1120]*(len(et)-70)
	# et.loc[et.RDepth > 1120, 'RDepth'] = 1120
	# et['fc'] = wp / 1000 + awc/1000
	# et['awc'] =awc/1000 #* (et['RDepth']+30) /230
	et["awc"] = awc
	et["wp"] = wp
	initIrriDate = -2
	if len(irri_record)>0:
		for item in irri_record:
			et.loc[ item["start"].split('T')[0].replace('-',''), 'IRRI'] = \
				et.loc[ item["start"].split('T')[0].replace('-',''), 'IRRI'] + item['volume']
		initIrriDate = np.max(np.nonzero(et.IRRI.tolist()))

	dep_thld_list = [dep_thld[0]]*25+ [dep_thld[1]]*35+ [dep_thld[2]]*40+ [dep_thld[3]]*500
	irri_re = np.zeros(len(et))
	for i in range(len(et)): # only calculating the first irrigating amount
		if (et.wp[i] + et.awc[i] *(1- dep_thld_list[i]/100)) > et.SM[i] and i>initIrriDate:
			irri_volume = (et.wp[i]+et.awc[i] * 0.9-et.SM[i])*et.RDepth[i]
			if irri_volume > 25:
				irri_volume = 25
			if (irri_volume < 5) or (i - initIrriDate < 3):
				# do not irrigate within 3 days after last irrigation
				# do not irrigate if suggested amount less than 5 mm
				continue
			irri_re[i] = irri_volume
			# #keep water depletion around 10% of fc
			initIrriDate = i
			break
	et['IRRIre'] = irri_re
	# et['IRRIre']=[0]*len(et)
	# et.loc[awc*(1-dep_thld)/2000+wp/2000>et['SM'], 'IRRIre'] = (awc/2000+wp/2000-et.SM[i])*et.RDepth


	et.to_csv(result_path + site_name+'_' + nc_file + '_'+ETorSM+'_IRRI_TABLE.csv')
	recom = et.loc[et.IRRIre>0, 'IRRIre']
	if len(recom)>0:
		print("Recommended irrigation volume: "+str(recom[0])+" mm on date: "+str(recom.index[0]) )
		return recom
	else:
		return 0

def buildKC(start, end, crop_type):
	kcCurve = {"Soybean":[0.4]*20+[i*0.75/34+0.4 for i in range(35)]+[1.15]*60+[i*(-0.65)/24+1.15 for i in range(25)]+[0.5]*300,
			   "Corn":[0.3]*25+[i*0.9/34+0.3 for i in range(35)]+[1.2]*40+[i*(-0.85)/34 + 1.2 for i in range(35)]+ [0.35]*300}
	# kc = {"Soybean": [1.0] * 20 + [1.15] * 90 + [0.5] * 25 + [0.1] * 300,
	# 	  "Corn": [1.0] * 30 + [1.2] * 90 + [0.6] * 50 + [0.1] * 300}
	delta = end -start
	day_num = delta.days-1 # day_num < growing season
	endD = end - dt.timedelta(days=1)
	index = pd.date_range(start, endD, freq='D')
	kcpd = pd.DataFrame(kcCurve[crop_type][:day_num+1],index = index, columns=[crop_type])
	return kcpd

def irriTable(planting_date, target_date, crop_type, irri_record, x,y,fc,wp,site_name, lon, lat, dep_thld,ETorSM):
	# planting_date = '03/31/2020'   2020050300
	# target_date = '12/30/2020'    2020050500
	# crop_type = "soybean"
	# irri_record = [
		# {"start": "2020-04-10T00:00:00.000Z", "end": "2020-04-10T00:00:00.000Z", "volume": 0.8, "changed": True},
		# {"start": "2020-04-08T00:00:00.000Z", "end": "2020-04-08T00:00:00.000Z", "volume": 0.6, "changed": True},
		# {"start": "2020-04-03T00:00:00.000Z", "end": "2020-04-03T00:00:00.000Z", "volume": 0.4, "changed": True}]
	# result_path = r'/home/hzhao/single-point/newoutput/JOHNSON/'
	result_path = r"/home/hzhao/single-point/newoutput/" + str(int(x)) + "/" + str(int(y))+"/"
	start = dt.datetime.strptime(str(planting_date), "%Y%m%d%H")
	end = dt.datetime.strptime(str(target_date), "%Y%m%d%H")
	nc_file = str(planting_date) + '.LDASOUT_DOMAIN1'
	# soil_info = getTexture(lat, lon)
	# wp = soil_info.iloc[0]['wp']#getWP(lat, lon)# 5.599 for Johnson
	# wp_value = wp
	# awc = soil_info.iloc[0]['fc'] - wp_value#getAWC(lat, lon)*2000
	wp_value = getWP(lat, lon)/100
	awc = getAWC(lat, lon)
	try:
		readVariable(nc_file, start, end, result_path, site_name)
	except Expection as e:
		logger.error(e, exc_info=True)
	if('ET' in ETorSM):
		return buildTable(start, end, nc_file, crop_type, irri_record, result_path, awc,wp_value, site_name,dep_thld, ETorSM)
	elif('SM' in ETorSM):
		return buildTableSM(nc_file, irri_record, result_path, awc, wp_value, site_name,dep_thld, ETorSM)
	else:
		print("Please indicate 'ET' or 'SM' in your command.")
		return 0

def test():
	para_kelly2020 = {'lon':-98.2034475, 'lat':41.9480136,'wp':0,'awc':0,'planting_date':'2020050100', 'target_date':'2020103123', 'irr_record':[],
					  'x':986, 'y':448, 'site_name':"UNL_KELLY_2020", 'dep_thld':[22,35,52,15],'ETorSM':"SM",'crop_type':"Corn"}

	result_path = r"/home/hzhao/single-point/newoutput/" + str(int(para_kelly2020['x'])) + "/" + str(int(para_kelly2020['y']))+"/"
	start = dt.datetime.strptime(str(para_kelly2020['planting_date']), "%Y%m%d%H")
	end = dt.datetime.strptime(str(para_kelly2020['target_date']), "%Y%m%d%H")
	nc_file = str(para_kelly2020['planting_date']) + '.LDASOUT_DOMAIN1'
	# readVariable(nc_file, start, end, result_path, para_kelly2020['site_name'], para_kelly2020['dep_thld'])
	getTexture(para_kelly2020['lat'],para_kelly2020['lon'])
if __name__ == '__main__':
	test()
