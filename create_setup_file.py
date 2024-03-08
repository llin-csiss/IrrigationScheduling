

import sys, time
import os
import numpy as np
import datetime as dt

import subprocess
import netCDF4 as nc
import requests
from osgeo import gdal, ogr, osr
from irriTable import transform_coordinate, get_raster_value


class Variables():

	def __init__(self, nc_file):
		self.nc_file = nc_file
		self.getVariable()

	def getVariable(self):
		nc_data = nc.Dataset(self.nc_file, 'r+')
		array_soilM = nc_data.variables['SMC'][:]  # soil moisture
		array_soilT = nc_data.variables['SOIL_T'][:]  # soil temperature
		array_st = nc_data.variables['TG'][:]  # ground temperature -- skin temperature
		array_lai = nc_data.variables['LAI'][:]  # LAI
		# sm = subprocess.check_output("ncdump -v SMC "+ self.nc_file)
		# print(sm)
		self.st =  array_st.reshape(-1)[0]
		self.lai = array_lai.reshape(-1)[0]
		self.soilT = array_soilT.reshape(-1)
		self.soilM = array_soilM.reshape(-1)

def CreateFile(locInfo, initStates, output):
	metadata = [
		"grid_id                           =  1 \n",
		"water_classification              = 17 \n",
		"urban_classification              = 13 \n",
		"ice_classification                = 15 \n",
		"land_cover_source                 = \"MODIFIED_IGBP_MODIS_NOAH\" \n",
	]

	conversion = [
		"have_relative_humidity            = .false. \n",
		"temperature_offset                = 273.15  \n",
		"temperature_scale                 = 1.0  \n",
		"pressure_scale                    = 100.0 \n",
		"precipitation_scale               = 0.014111 \n",
	]

	file = open(output, "w")

	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("!Location Information\n")
	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("&location\n")
	file.writelines(locInfo)
	file.write("/\n")

	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("!Initial States\n")
	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("&initial\n")
	file.writelines(initStates)
	file.write("/\n")

	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("!Metadata\n")
	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("&metadata\n")
	file.writelines(metadata)
	file.write("/\n")

	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("!Conversion information\n")
	file.write("!---------------------------------------------------------------------------------------------------------\n")
	file.write("&conversion\n")
	file.writelines(conversion)
	file.write("/\n")

	file.write("----------------------------------------------------------------------------------------------------------------\n")
	file.write("   date/time     | windspeed  | temperature | humidity | pressure | shortwave | longwave | precipitation\n")
	file.write("yyyy mm dd hh mi |  m s{-1}   |     C       |    %     |    mb    |  W m{-2}  |  W m{-2} | Inches/timestep\n")
	file.write("-----------------------------------------------------------------------------------------------------------------\n")

	file.close()

def getElevation(lat, lon):
	# pload = {'x':str(lon), 'y':str(lat), 'units':'Meters', 'output':'json'}
	# # host = 'https://nationalmap.gov/epqs/pqs.php'
	# host='https://epqs.nationalmap.gov/v1/json'
	# r = requests.post(url=host, data=pload)
	# result = r.json()
	# print(result)
	# elevation =result['value']
	elevation = 620.547
	return(elevation)

# def getTexture(lat, lon):
# 	soil_texture = subprocess.check_output("gdallocationinfo -valonly -wgs84 /home/hzhao/sample/soilProps/texture_500m_remap.tif "+str(lon)+" "+str(lat), shell=True)
# 	return str(soil_texture).split("'")[1].split("\\")[0]

def getTexture(lat, lon):
	texture_path = "./soilProps/"
	filename = texture_path+"texture_500m_remap.tif"
	dataset = gdal.Open(filename)
	geo_x, geo_y = transform_coordinate(lat, lon)
	texture_value = get_raster_value(geo_x, geo_y, dataset)
	return str(texture_value)


def CreateLDASIN(input, output, x, y, rainvalue):
	#os.system("ncks -d south_north," + y + "," + y + " -d west_east," + x + "," + x + " " + input + " " + output)
	os.system("ncks -d south_north,"+y+","+y+" -d west_east,"+x+","+x+" "+input+" "+output)
	os.system("ncap2 -O -s 'rainrate_nwm=rainrate_nwm+"+ str(rainvalue)+"' "+output+" " +output)
	os.system("ncrename -v t2d_nwm,T2D " +output)
	os.system("ncrename -v q2d_nwm,Q2D " +output)
	os.system("ncrename -v psfc_nwm,PSFC " +output)
	os.system("ncrename -v u2d_nwm,U2D " +output)
	os.system("ncrename -v v2d_nwm,V2D " +output)
	os.system("ncrename -v lwdown_nwm,LWDOWN " +output)
	os.system("ncrename -v swdown_nwm,SWDOWN " +output)
	os.system("ncrename -v rainrate_nwm,RAINRATE " +output)
	os.system("chmod +x "+ output)
	#os.system("rm temp2.nc")


def CreateDat( lon, lat, startDate, x,y):


	nc_file =r'/home/hzhao/single-point/analysis/'+str(int(x)) + '/' + str(int(y))+'/RESTART.'+str(startDate)+'_DOMAIN1'
	nc_variables = Variables(nc_file)

	# lat = 41.1651
	# lon = -96.4766
	veg_cate = 12
	soil_cate = getTexture(lat,lon)
	deep_soil_t = 285.0
	elevation = getElevation(lat,lon)
	sea_ice = 0.0
	max_veg_pct = 96.0
	min_veg_pct = 1.0
	land_mask = 1
	nsoil = 4
	locationInfo = [
		' latitude                   =  ' + str(lat) + '\n',
		' longitude                  =  ' + str(lon) + '\n',
		' vegetation_category        =  ' + str(veg_cate) + '\n',
		' soil_category              =  ' + str(soil_cate) + '\n',
		' deep_soil_temperature      =  ' + str(deep_soil_t) + '\n',
		' elevation                  =  ' + str(elevation) + '\n',
		' sea_ice                    =  ' + str(sea_ice) + '\n',
		' maximum_vegetation_pct     =  ' + str(max_veg_pct) + '\n',
		' minimum_vegetation_pct     =  ' + str(min_veg_pct) + '\n',
		' land_mask                  =  ' + str(land_mask) + '\n',
		' nsoil                      =  ' + str(nsoil) + '\n'
	]

	snow_depth = 0
	snow_water_equivalent = 0
	canopy_water = 0
	skin_temperature = nc_variables.st
	soil_layer_thickness = [0.10, 0.30, 0.60, 1.00]
	soil_layer_nodes = [0.05, 0.25, 0.70, 1.50]
	soil_temperature = nc_variables.soilT
	soil_moisture = nc_variables.soilM
	leaf_area_index = nc_variables.lai + 0.1
	initialStates = [
		'snow_depth                 = ' + str(snow_depth) + '\n',
		'snow_water_equivalent      = ' + str(snow_water_equivalent) + '\n',
		'canopy_water               = ' + str(canopy_water) + '\n',
		'skin_temperature           = ' + str(skin_temperature) + '\n',
		'soil_layer_thickness       = ' + ',    '.join(str(e) for e in soil_layer_thickness) + '\n',
		'soil_layer_nodes           = ' + ',    '.join(str(e) for e in soil_layer_nodes) + '\n',
		'soil_temperature           = ' + ',    '.join(str(e) for e in soil_temperature) + '\n',
		'soil_moisture              = ' + ',    '.join(str(e) for e in soil_moisture) + '\n',
		'leaf_area_index            = ' + str(leaf_area_index) + '\n',
	]

	output = r'/home/hzhao/single-point/workshop/bondville.dat'
	CreateFile(locationInfo, initialStates, output)
	print("Bondville file is created successfully!")


def test():
	nc_variables = Variables(r'C:\Users\nuds\Dropbox\singlepoint/RESTART.2020040100_DOMAIN1')

	lat = 41.1651
	lon = -96.4766
	veg_cate = 12
	soil_cate = 8
	deep_soil_t = 285.0
	elevation = 361.0
	sea_ice = 0.0
	max_veg_pct = 96.0
	min_veg_pct = 1.0
	land_mask = 1
	nsoil = 4
	locationInfo = [
		' latitude                   =  '+str(lat)+'\n',
		' longitude                  =  ' + str(lon)+ '\n',
		' vegetation_category        =  ' +str(veg_cate)+'\n',
		' soil_category              =  ' + str(soil_cate) + '\n',
		' deep_soil_temperature      =  ' +str(deep_soil_t) + '\n',
		' elevation                  =  ' +str(elevation) + '\n',
		' sea_ice                    =  ' +str(sea_ice) + '\n',
		' maximum_vegetation_pct     =  ' +str(max_veg_pct) + '\n',
		' minimum_vegetation_pct     =  ' +str(min_veg_pct) + '\n',
		' land_mask                  =  ' +str(land_mask) + '\n',
		' nsoil                      =  ' +str(nsoil) + '\n'
	]


	snow_depth = 0
	snow_water_equivalent = 0
	canopy_water = 0
	skin_temperature = nc_variables.st
	soil_layer_thickness = [0.10,0.30,0.0,1.00]
	soil_layer_nodes = [0.05,0.25,0.70,1.50]
	soil_temperature = nc_variables.soilT
	soil_moisture = nc_variables.soilM
	leaf_area_index = nc_variables.lai+0.1
	initialStates = [
		'snow_depth                 = ' + str(snow_depth) + '\n',
		'snow_water_equivalent      = ' + str(snow_water_equivalent) + '\n',
		'canopy_water               = ' + str(canopy_water) + '\n',
		'skin_temperature           = ' + str(skin_temperature) + '\n',
		'soil_layer_thickness       = ' + ',    '.join(str(e) for e in soil_layer_thickness) + '\n',
		'soil_layer_nodes           = ' + ',    '.join(str(e) for e in soil_layer_nodes) + '\n',
		'soil_temperature           = ' + ',    '.join(str(e) for e in soil_temperature) + '\n',
		'soil_moisture              = ' + ',    '.join(str(e) for e in soil_moisture) + '\n',
		'leaf_area_index            = ' + str(leaf_area_index) + '\n',
	]


	CreateFile(locationInfo, initialStates)

def main():
	lat = 41.1651
	lon = -96.4766
	nc_file =r'/home/hzhao/single-point/analysis/48/471/RESTART.2020050300_DOMAIN1'
	elevation = getElevation(lat,lon)
	# soil_texture = getTexture(lat,lon)
	print(elevation)

if __name__ == "__main__":
	main()