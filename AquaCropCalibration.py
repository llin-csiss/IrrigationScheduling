import numpy as np
import pandas as pd
import os
from os.path import exists

def getACsm(AC_sm_file,AC_rd_file):
	AC_sm = pd.read_csv(AC_sm_file)
	AC_sm = AC_sm.loc[AC_sm['Wr'] > 0]
	AC_rd = pd.read_csv(AC_rd_file)
	AC_rd = AC_rd.loc[AC_rd['z_root'] > 0]
	AC_sm['SM'] = AC_sm['Wr']/(1000*AC_rd['z_root'])
	return AC_sm['SM']

def getSM(site):
	filePath = r"/home/hzhao/single-point/newoutput/" + site[2] + "/" + site[3]
	nc_file = site[1]+'.LDASOUT_DOMAIN1'
	output = filePath+'/UNL_' + site[0]+'_AquaCrop_SM.csv'
	AC_rainfed_sm_file = filePath + '/UNL_' + site[0] + '_recommended schedule_TEST_daily_water_flux.csv'
	AC_real_irr_sm_file = filePath + '/UNL_' + site[0] + '_real schedule_daily_water_flux.csv'
	AC_real_irr_rd_file = filePath + '/UNL_' + site[0] + '_real schedule_crop_growth.csv'
	AC_rainfed_rd_file = filePath + '/UNL_' + site[0] + '_recommended schedule_TEST_crop_growth.csv'
	HM_rainfed_sm_file = filePath + '/UNL_' + site[0] + '_'+nc_file+'_ET.csv'
	AC_rainfed_sm = getACsm(AC_rainfed_sm_file,AC_rainfed_rd_file)
	AC_real_irr_sm = getACsm(AC_real_irr_sm_file, AC_real_irr_rd_file)

	HM_rainfed_sm = pd.read_csv(HM_rainfed_sm_file)
	HM_sm = HM_rainfed_sm['SM'][:len(AC_rainfed_sm)]
	# AC_rainfed_sm.index = HM_sm.index
	# AC_real_irr_sm.index = HM_sm.index
	HM_sm.columns=['SM_HRLDAS']

	AC_sm = AC_rainfed_sm.to_frame().join(AC_real_irr_sm, lsuffix='_AquaCrop',rsuffix='_real_irr')
	sm = HM_sm.to_frame().join(AC_sm)
	sm.columns=['SM_HRLDAS','SM_AquaCrop','SM_real_irr']

	sm.to_csv(output)
	return sm

def getAquaCropSM():
	siteNames = [['EAST_2019', '2019050200', '989', '447'], ['JOHNSON_2020', '2020042500', '984', '466'],
				 ['HOME_2019', '2019050400', '987', '447'], ['HOME_2020', '2020050100', '987', '447'],
				 ['KELLY_2019', '2019042500', '986', '448'], ['KELLY_2020', '2020050100', '986', '448'],
				 ['LINKS_2019', '2019042400', '984', '450'],['NORTH_2020', '2020050800', '987', '448']]
	sm=pd.DataFrame([])
	for i,site in enumerate(siteNames):
		print(f"######################################{site[0]}")
		if sm.empty:
			sm = getSM(site)
		else:
			temp = getSM(site)
			rsuf = '_'+siteNames[i][0] if i==len(siteNames)-1 else None
			sm = pd.merge(sm, temp, how='outer',right_index=True, left_index=True,suffixes=('_'+siteNames[i-1][0],rsuf))

	sm.to_csv(r"/home/hzhao/single-point/newoutput/AquaCrop_SM.csv")

if __name__ == '__main__':	# lat = 41.9406
	getAquaCropSM()