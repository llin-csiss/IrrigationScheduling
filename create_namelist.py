import os
import sys, time
import datetime as dt
import numpy as np

def createNamelist(file_path, lines):
	file = open(file_path, 'w')

	file.write('\n&NOAHLSM_OFFLINE\n\n')
	file.writelines(lines)
	file.close()


def getContent(x, y,start,end):
	setup_path = r"/home/hzhao/single-point/workshop/hrldas_setup_single_point.nc"
	indir = r"/home/hzhao/single-point/analysis/"+str(int(x))+"/"+str(int(y))
	outdirX = r"/home/hzhao/single-point/newoutput/"+str(int(x))
	outdir = outdirX + '/' + str(int(y))
	if not os.path.isdir(outdirX):
		os.makedirs(outdirX)
	if not os.path.isdir(outdir):
		os.makedirs(outdir)
	kdays =abs(dt.datetime.strptime(str(end),'%Y%m%d%H') - dt.datetime.strptime(str(start),'%Y%m%d%H')).days

	dyna_veg = 5
	crop_option = 1

	lines = [
		"HRLDAS_SETUP_FILE = \""+ setup_path+"\"\n",
		"INDIR = \""+indir+"\"\n",
		"OUTDIR = \""+outdir+"\"\n\n",

		"START_YEAR = " + str(start)[:4]+"\n",
		"START_MONTH = " + str(start)[4:6]+"\n",
		"START_DAY = "+str(start)[6:8] + "\n",
		"START_HOUR = " + str(start)[8:10] + "\n",
		"START_MIN = 00\n\n",

		"KDAY = " + str(kdays) +"\n",
		"SPINUP_LOOPS = 0\n\n",

		"DYNAMIC_VEG_OPTION                    = " + str(dyna_veg) + "\n",
		"CANOPY_STOMATAL_RESISTANCE_OPTION     = 1\n",
		"BTR_OPTION                            = 1\n",
		"RUNOFF_OPTION                         = 3\n",
		"SURFACE_DRAG_OPTION                   = 1\n",
		"FROZEN_SOIL_OPTION                    = 1\n",
		"SUPERCOOLED_WATER_OPTION              = 1\n",
		"RADIATIVE_TRANSFER_OPTION             = 3\n",
		"SNOW_ALBEDO_OPTION                    = 1\n",
		"PCP_PARTITION_OPTION                  = 1\n",
		"TBOT_OPTION                           = 2\n",
		"TEMP_TIME_SCHEME_OPTION               = 1\n",
		"GLACIER_OPTION                        = 1\n",
		"SURFACE_RESISTANCE_OPTION             = 1\n",
		"SOIL_DATA_OPTION                      = 1\n",
		"PEDOTRANSFER_OPTION                   = 1\n",
		"CROP_OPTION                           = " + str(crop_option) + "\n",
		"IRRIGATION_OPTION                     = 0\n",
		"IRRIGATION_METHOD                     = 0\n\n",

		"FORCING_TIMESTEP                      = 3600\n",
		"NOAH_TIMESTEP                         = 3600\n",
		"OUTPUT_TIMESTEP                       = 3600\n\n",

		"SPLIT_OUTPUT_COUNT                    = 0\n",
		"SKIP_FIRST_OUTPUT                     = .false.\n",
		"RESTART_FREQUENCY_HOURS               = 0\n\n",

		"NSOIL                                 = 4\n",
		"soil_thick_input(1)                   = 0.10\n",
		"soil_thick_input(2)                   = 0.30\n",
		"soil_thick_input(3)                   = 0.60\n",
		"soil_thick_input(4)                   = 1.00\n\n",

		"ZLVL                                  = 10.0\n\n",
		"SF_URBAN_PHYSICS                      = 0\n",
		"USE_WUDAPT_LCZ                        = 0\n",

		"\n/\n"
	]
	return lines

def createFile(x,y,start,end):
	file_path = r'/home/hzhao/single-point/newrun/hrldas/run/namelist.hrldas'
	test_path = r'C:\Users\nuds\Dropbox\singlepoint/namelist_test.hrldas'
	lines = getContent(x, y, start, end)
	createNamelist(file_path, lines)

def main():
	x = 48
	y = 471
	start = 2020050300
	end = 2020050600
	createFile(x,y,start,end)

if __name__ == "__main__":
	main()