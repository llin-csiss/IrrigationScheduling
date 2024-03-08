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
from scipy.optimize import fmin
from tqdm.notebook import tqdm  # progress bar
import re
import time

xy_dict = {"UNL_JOHNSON_2020": ['984','466'], "UNL_KELLY_2020": ['986','448'], "UNL_HOME_2020": ['987','447'],
				 "UNL_NORTH_2020": ['987','448'], "UNL_HOME_2019": ['987','447'], "UNL_EAST_2019": ['989','447'],
				 "UNL_KELLY_2019": ['986','448'], "UNL_LINKS_2019": ['984','450']
				 }

def run_model(smts, site_name, ETorSM):
	"""
	funciton to run model and return results for given set of soil moisture targets
	"""
	#site_name = 'UNL_JOHNSON_2020'
	result_path = '/home/hzhao/single-point/newoutput/' + xy_dict[site_name][0] + '/' + xy_dict[site_name][1] + '/'
	cmd = "java -jar ids-1.0.0-beta2-SNAPSHOT_UNL_ETSM_4threshold.jar " + site_name + " " + str(smts[0]) + " " + str(
		smts[1]) + " " + str(smts[2]) + " " + str(smts[3]) + " "+ETorSM
	print(cmd)
	time_start = time.perf_counter()
	p = subprocess.Popen(cmd, shell=True)
	p.wait()
	time_end = time.perf_counter()
	print('time cost: ', time_end-time_start, 's')

	out = pd.read_csv(result_path+site_name+'_'+ETorSM+'_4threshold_yield_estimates.csv')
	y = out.loc[out['label']=='recommended schedule_'+ETorSM+'_'+'4threshold']

	# maize = Crop('Maize',planting_date='05/01') # define crop
	# loam = Soil('ClayLoam') # define soil
	# init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions
	#
	# irrmngt = IrrigationManagement(irrigation_method=1,SMT=smts,MaxIrrSeason=max_irr_season) # define irrigation management
	#
	# # create and run model
	# model = AquaCropModel(f'{year1}/05/01',f'{year2}/10/31',wdf,loam,maize,
	# 					  irrigation_management=irrmngt,initial_water_content=init_wc)
	#
	# model.run_model(till_termination=True)
	# return model.get_simulation_results()
	return y

def evaluate(smts,site_name,ETorSM,test=False):
	"""
	funciton to run model and calculate reward (yield) for given set of soil moisture targets
	"""
	# run model
	out = run_model(smts, site_name, ETorSM)
	# get yields and total irrigation
	yld = out['Yield (tonne/ha)']
	tirr = out['Seasonal irrigation (mm)']

	reward=yld

	# return either the negative reward (for the optimization)
	# or the yield and total irrigation (for analysis)
	if test:
		return yld,tirr,reward
	else:
		return -reward


def get_starting_point(num_smts,site_name,ETorSM, num_searches):
	"""
	find good starting threshold(s) for optimization
	"""

	# get random SMT's
	x0list = np.random.randint(100, size=(num_searches, num_smts))
	rlist = []
	# evaluate random SMT's
	for xtest in x0list:
		print('======================\n try to get starting point')
		r = evaluate(xtest,site_name,ETorSM )
		rlist.append(r)

	# save best SMT
	x0 = x0list[np.argmin(rlist)]

	return x0

def optimize(num_smts,site_name,ETorSM,num_searches=100):
	"""
	optimize thresholds to be profit maximising
	"""
	# get starting optimization strategy

	x0=get_starting_point(num_smts,site_name, ETorSM,num_searches)
	# run optimization
	res = fmin(evaluate, x0, xtol=1, ftol=0.0001, disp=0, args=(site_name,ETorSM))
	# reshape array
	smts= res.squeeze()
	# evaluate optimal strategy
	return smts



def yieldOpt(site_name, ETorSM):
	# opt_smts = []
	# yld_list = []
	# tirr_list = []
	result_path = '/home/hzhao/single-point/newoutput/' + xy_dict[site_name][0] + '/' + xy_dict[site_name][1] + '/'
	# for max_irr in tqdm(range(0, 500, 50)):
		# find optimal thresholds and save to list
	smts = optimize(4, site_name, ETorSM)
	np.savetxt(result_path+site_name+'_'+ETorSM+'_4threshold.csv', smts, delimiter=",",fmt='%s')
	#opt_smts.append(smts)
		# save the optimal yield and total irrigation
	#yld, tirr, _ = evaluate(smts,site_name, True)
	return smts
	# 	yld_list.append(yld)
	# 	tirr_list.append(tirr)
	#
	# # create plot
	# fig,ax=plt.subplots(1,1,figsize=(13,8))
	#
	# # plot results
	# ax.scatter(tirr_list,yld_list)
	# ax.plot(tirr_list,yld_list)
	#
	# # labels
	# ax.set_xlabel('Total Irrigation (ha-mm)',fontsize=18)
	# ax.set_ylabel('Yield (tonne/ha)',fontsize=18)
	# ax.set_xlim([-20,600])
	# ax.set_ylim([2,15.5])
	#
	# # annotate with optimal thresholds
	# bbox = dict(boxstyle="round",fc="1")
	# offset = [15,15,15, 15,15,-125,-100,  -5, 10,10]
	# yoffset= [0,-5,-10,-15, -15,  0,  10,15, -20,10]
	# for i,smt in enumerate(opt_smts):
	# 	smt=smt.clip(0,100)
	# 	ax.annotate('(%.0f, %.0f, %.0f, %.0f)'%(smt[0],smt[1],smt[2],smt[3]),
	# 				(tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
	# 				bbox=bbox,fontsize=12)

def main():
	site_list = ["UNL_JOHNSON_2020", "UNL_KELLY_2020", "UNL_HOME_2020",
			   "UNL_NORTH_2020", "UNL_HOME_2019", "UNL_EAST_2019",
			   "UNL_KELLY_2019", "UNL_LINKS_2019"
			   ]
	ETorSM_options = ["ET","SM", "ET_F0", "SM_F0"]
	opt_smts = pd.DataFrame([], index = ETorSM_options, columns=site_list)
	for i in site_list:
		for j in ETorSM_options:
			smts = yieldOpt(i,j)
			opt_smts.loc[j,i] = smts
	print(opt_smts)
	opt_smts.to_csv('/home/hzhao/single-point/newoutput/4thresholds.csv')
	# for i in site_list:
	# 	yieldOpt(i,"SM")

def run_default():
	site_list = ["UNL_JOHNSON_2020", "UNL_KELLY_2020", "UNL_HOME_2020",
			   "UNL_NORTH_2020", "UNL_HOME_2019", "UNL_EAST_2019",
			   "UNL_KELLY_2019", "UNL_LINKS_2019"
			   ]
	ETorSM_options = ["ET1","SM1"]
	# opt_smts = pd.DataFrame([], index = ETorSM_options, columns=site_list)
	smts=[50,50,50,50]
	for i in site_list:
		for j in ETorSM_options:
			run_model(smts,i,j)

if __name__ == '__main__':
	main()