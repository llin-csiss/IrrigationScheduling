import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import datetime as dt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
from os.path import exists
import re

from validation import bias, rmse, ubrmse

siteNames = ['EAST_2019', 'HOME_2019', 'HOME_2020', 'JOHNSON_2020', 'KELLY_2019', 'KELLY_2020', 'LINKS_2019',
			 'NORTH_2020']
thldList = range(30, 81,5)#['30', '35', '40', '44', '49', '54', '60', '65', '70', '75', '80']
thldList = ['4threshold']
def main():
	resultPath = r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m'
	yieldFile = resultPath+'/yield_ET_4threshold.csv'

	if not exists(yieldFile):
		arrangeData(resultPath, yieldFile)
	else:
		#for site in siteNames:
		#	plotLine(resultPath, site)
		plotLine(resultPath, 'NORTH_2020')
def arrangeData(filePath, yieldFile):
	# siteNames = ['EAST_2019','HOME_2019','HOME_2020','JOHNSON_2020','KELLY_2019','KELLY_2020','LINKS_2019','NORTH_2020']
	# thldList=['30','35','40','44','49','54','60','65','70','75','80']
	ydict ={'Yield (tonne/ha)':[],'Seasonal irrigation (mm)':[],'label':[],'site':[]}
	ypd = pd.DataFrame(ydict)
	for site in siteNames:
		print(site)
		for thld in thldList:
			file = filePath +'/'+site.replace('_','')+'/UNL_'+site+'_ET_'+str(thld)+'_yield_estimates.csv'
			y = pd.read_csv(file)
			ye = y.loc[y['label']=='recommended schedule_ET_'+str(thld)]
			ye['site'] = site
			yr = y.loc[y['label']=='real schedule']
			yr['site'] = site
			ypd = ypd.append(yr, ignore_index=True)
			ypd = ypd.append(ye, ignore_index=True)
	ypd = ypd.drop_duplicates()
	ypd = ypd.iloc[:,:-1]
	ypd.to_csv(yieldFile)

def plotLine(resultPath, site):
	yld = pd.read_csv(resultPath+'/yield_ET.csv')
	#yld = yld.sort_values('Seasonal irrigation (mm)')
	tirr_list = yld.loc[yld['site'] == site,'Seasonal irrigation (mm)'].tolist()
	yld_list = yld.loc[yld['site'] == site,'Yield (tonne/ha)'].tolist()
	thd_list = yld.loc[yld['site'] == site,'label'].tolist()
	# create plot
	fig, ax = plt.subplots(1, 1, figsize=(13, 8))

	# plot results
	ax.scatter(tirr_list, yld_list)
	ax.plot(tirr_list, yld_list)

	# labels
	ax.set_xlabel('Total Irrigation (ha-mm)',fontsize=18)
	ax.set_ylabel('Yield (tonne/ha)',fontsize=18)
	ax.set_title(site, fontsize=24)

	# annotate with optimal thresholds
	bbox = dict(boxstyle="round", fc="1")
	offset = [15, 15, -35, -25, -15, 15, 15, 15, 15, 15, 15, 15]
	yoffset = [0, 5, -20, 10, 10, -10, -10, 0, 0, 0, 0, 0]
	for i, smt in enumerate(thd_list):
		#smt = smt.clip(0, 100)
		ax.annotate('(%s)' % (smt.replace('recommended schedule_ET_','MAD ')),
					(tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
					bbox=bbox, fontsize=12)
	plt.savefig(resultPath+'/pic/yield_MAD_'+site+'.png')
	#plt.show()


def plotThreshold(resultPath, filename):

	threshold = pd.read_csv(resultPath+'/'+filename,index_col=0)
	t2020 = threshold.iloc[:,:4]
	t2019 = threshold.iloc[:,4:8]
	print(t2020)
	plt.rc('font', size=12)
	fig,((ax1),(ax2)) = plt.subplots(2,1, figsize=(13, 8))
	#ax2 = plt.subplots(212, figsize=(13, 8))
	t2019.plot(ax=ax1)
	t2020.plot(ax=ax2)
	ax1.set_title('thresholds for 4 stages(%) - 2019')
	ax2.set_title('thresholds for 4 stages(%) - 2020')
	ax1.legend()
	ax2.legend()
	plt.gcf().set_size_inches(20,10)
	plt.tight_layout()
	plt.savefig(resultPath+'/'+filename.split('.')[0]+'.png',dpi=100)
	plt.show()


def smooth(scalars, weight):  # Weight between 0 and 1
	last = scalars[0]  # First value in the plot (first timestep)
	smoothed = list()
	for point in scalars:
		smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
		smoothed.append(smoothed_val)  # Save it
		last = smoothed_val  # Anchor the last smoothed value

	return smoothed

def getNetReturn(file,name,resultPath, site):
	filePath = resultPath+'/'+site.replace('_','')+'/'+file
	file_pd = pd.read_csv(filePath)
	file_pd.columns = [name]
	pd_smoothed = pd.DataFrame(smooth(file_pd.values, 0.9), columns=['{}_smoothed'.format(name)])
	file_pd = pd.merge(file_pd, pd_smoothed, how='left', right_index=True, left_index=True)
	return file_pd

def plotNetReturn():
	resultPath = r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m'
	site = 'KELLY_2020'
	dql65R1 = ['UNL_'+site+'_DQL_0.65_R1_net_return.csv', 'DQL-alpha0.65-R1']
	# dql65R0 = ['UNL_'+site+'_DQL_0.65_R0_H0_net_return.csv', 'DQL-alpha0.65-R0']
	# ql65R0 = ['UNL_'+site+'_QL_0.65_R0_net_return.csv','QL-alpha0.65-R0']
	ql65R1 = ['UNL_'+site+'_QL_0.65_R1_net_return.csv','QL-alpha0.65-R1']
	ql65R1_b = 93.139
	dql65R1_b = 170

	# site = 'HOME_2019'
	# ql65R1 = ['UNL_KELLY_2019_QL_R1_H0_net_return.csv','QL-alpha0.65-R1']
	# dql65R1 = ['UNL_JOHNSON_2020_DQL_R1_H0_net_return.csv', 'DQL-alpha0.65-R0']
	# ql65R1_b = 51.585
	# dql65R1_b = 400

	dql65R1_pd = getNetReturn(dql65R1[0],dql65R1[1],resultPath,site)
	# dql65R0_pd = getNetReturn(*dql65R0)
	# ql65R0_pd = getNetReturn(*ql65R0)
	ql65R1_pd = getNetReturn(ql65R1[0],ql65R1[1],resultPath,site)
	dql_pd = pd.merge(dql65R1_pd, ql65R1_pd, how='left', right_index=True, left_index=True)

	# dql_pd = pd.merge(dql65R1_pd, ql65R1_pd, how='left', right_index=True, left_index=True)
	# dql_pd = pd.merge(dql_pd, ql65R0_pd, how='left', right_index=True, left_index=True)
	# dql_pd = pd.merge(dql_pd, dql65R0_pd, how='left', right_index=True, left_index=True)
	dql_pd = dql_pd.fillna(method='ffill')

	plt.rcParams.update({'font.size': 18})
	fig, ax = plt.subplots(1, 1, figsize=(13, 8))

	# compare net return with and without forecasted rainfall in q-learning model
	# ax.plot(range(len(dql_pd)), dql_pd[ql65R0[1]], 'C0', alpha=0.2,)
	# ax.plot(range(len(dql_pd)), dql_pd[ql65R1[1]], 'C1', alpha=0.2)
	# ax.plot(range(len(dql_pd)), dql_pd['{}_smoothed'.format(ql65R0[1])], 'C0', label=ql65R0[1])
	# ax.plot(range(len(dql_pd)), dql_pd['{}_smoothed'.format(ql65R1[1])], 'C1', label=ql65R1[1])

	# compare net return with and without forecasted rainfall in deep q-learning model
	# ax.plot(range(len(dql_pd)), dql_pd[dql65R0[1]], 'C0', alpha=0.2,)
	# ax.plot(range(len(dql_pd)), dql_pd[dql65R1[1]], 'C1', alpha=0.2)
	# ax.plot(range(len(dql_pd)), dql_pd['{}_smoothed'.format(dql65R0[1])], 'C0', label=dql65R0[1])
	# ax.plot(range(len(dql_pd)), dql_pd['{}_smoothed'.format(dql65R1[1])], 'C1', label=dql65R1[1])

	# Compare between QL and DQL
	# ax.plot(range(len(dql_pd)), [i + ql65R1_b for i in dql_pd[ql65R1[1]]], 'C0', alpha=1,label='Q Learning')
	# ax.plot(range(len(dql_pd)), [a + (-200 if i <338 else 400) for i,a in enumerate(dql_pd[dql65R1[1]])], 'C1', alpha=1, label='Deep Q Network')
	ax.plot(range(len(dql_pd)), [i + ql65R1_b for i in dql_pd[ql65R1[1]]], 'C0', alpha=1, label='Q Learning')
	ax.plot(range(len(dql_pd)), [i + dql65R1_b for i in dql_pd[dql65R1[1]]], 'C1',
			alpha=1, label='Deep Q Network')
	# ax.plot(range(len(dql_pd)), dql_pd['{}_smoothed'.format(ql65R1[1])], 'C0', label=ql65R1[1])
	# ax.plot(range(len(dql_pd)), dql_pd['{}_smoothed'.format(dql65R1[1])], 'C1', label=dql65R1[1])
	ax.legend()
	ax.set_xlabel('Episode')
	ax.set_ylabel('Net Return')

	plt.gcf().set_size_inches(10, 5)
	plt.tight_layout()
	plt.savefig(r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m' + '/' + site.replace('_',
																							'') + '/'+site+'net_return_R1.png',
				dpi=100)

	plt.show()

def plotAquaCropSM():
	AquaCropSM = r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m/AquaCrop_SM.csv'
	siteNames = [['EAST_2019', '2019050200', '989', '447'], ['JOHNSON_2020', '2020042500', '984', '466'],
				 ['HOME_2019', '2019050400', '987', '447'], ['HOME_2020', '2020050100', '987', '447'],
				 ['KELLY_2019', '2019042500', '986', '448'], ['KELLY_2020', '2020050100', '986', '448'],
				 ['LINKS_2019', '2019042400', '984', '450'],['NORTH_2020', '2020050800', '987', '448']]
	AquaCropSM_pd = pd.read_csv(AquaCropSM, index_col=0)
	rmse_list = []
	bias_list = []
	ubrmse_list = []
	r2_list=[]

	plt.rc('font', size=12)
	fig, axs = plt.subplots(4,2)
	for i in range(8):

		pred = AquaCropSM_pd.iloc[:,i*3].to_numpy().astype(float)
		tar = AquaCropSM_pd.iloc[:,i*3+1].to_numpy().astype(float)

		bias_value = bias(pred, tar)
		rmse_value = rmse(pred, tar)
		ubrmse_value = ubrmse(pred, tar)
		bias_list.append(bias_value)
		rmse_list.append(rmse_value)
		ubrmse_list.append(ubrmse_value)
		AquaCropSM_pd['SM_'+siteNames[i][0]] = pred - bias_value
		AquaCropSM_pd['SM_' + siteNames[i][0]].plot(ax=axs[0+i//2,i%2])
		AquaCropSM_pd.iloc[:, i * 3 + 1].plot(ax=axs[0 + i // 2, i % 2])
		axs[0 + i // 2, i % 2].set_title('RMSE={}'.format(ubrmse_value))
		# axs[0 + i // 2, i % 2].title.set_size(6)
		axs[0 + i // 2, i % 2].legend(fontsize=12)

	#fig, axs = plt.subplots(4,2)
	for i in range(8):
		pred = AquaCropSM_pd.loc[:, 'SM_' + siteNames[i][0]].to_numpy().astype(float)
		tar = AquaCropSM_pd.iloc[:, i * 3 + 1].to_numpy().astype(float)
		d1 = tar[:, np.newaxis]
		d2 = pred[:, np.newaxis]
		lrModel = LinearRegression()
		lrModel.fit(d1, d2)
		predicts = lrModel.predict(d1)
		R2 = lrModel.score(d1, d2)
		r2_list.append(R2)
		print('R2 = %.2f' % R2)
		# coef = lrModel.coef_
		# intercept = lrModel.intercept_
		# axs[0 + i // 2, i % 2].plot(tar,pred,'o')
		# axs[0 + i // 2, i % 2].axline((0,0),slope=1)
		# axs[0 + i // 2, i % 2].set_title('R^2={}'.format(R2))
		# axs[0 + i // 2, i % 2].title.set_size(6)
		# axs[0 + i // 2, i % 2].plot(d1, predicts, color='red', label='predicted value')
		# # axs[0 + i // 2, i % 2].set(title=f'y={coef[0][0]}*x+{intercept[0]} with R2={R2}')
	metric_pd = pd.DataFrame({'Bais':bias_list,'RMSE':rmse_list,'ubRMSE':ubrmse_list,'R2':r2_list})
	metric_pd.to_csv(r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m/AquaCrop_SM_metrics.csv')
	plt.gcf().set_size_inches(20,10)
	plt.tight_layout()
	plt.savefig(r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m/AquaCrop_SM.png',dpi=100)
	plt.show()

def plotYield():
	siteNames = ['EAST_2019','HOME_2019', 'KELLY_2019', 'LINKS_2019', 'NORTH_2020','HOME_2020','KELLY_2020','JOHNSON_2020']
	total_irr_actual = [181.6,191.5,206.8,198.1,251.0,278.4,278.4,276.4]

	yield_ET_R0 = [13.85,14.2,13.3,12.9,13.3,13.8,13.9,13.7]
	yield_actual = [13.5,13.9,13.4,12.8,13.3,13.6,13.7,13.9]
	yield_SM_R0 = [13.8,14.1,13.3,12.8,13.3,13.7,13.9,13.8]
	yield_ET_R1 = [13.6,14.1,13.5,12.8,13.3,13.7,13.6,13.7]
	yield_SM_R1 = [13.6,14.0,13.5,12.8,13.3,13.6,13.6,13.7]
	yield_QL_R1 =[14.025]
	total_irr_QL_R1 = [265.5]
	total_irr_SM_R0 = [137.6,133.3,129.5,147.6,215,245.1,254.9,256.4]
	total_irr_ET_R0 = [149.3,146.1,137.6,157.6,219.1,245.6,249,244.8]
	total_irr_ET_R1 = [119.2,95.9,130.3,134.4,199.2,225,226.7,216.1]
	total_irr_SM_R1 = [114.6,90.3,121.8,129.4,195.2,220.4,222.5,208.9]
	columns = ['IA',
			   'IA_ET_R0',
			   'IA_SM_R0',
			   'IA_ET_R1',
			   'IA_SM_R1',
			   ]
	yield_pd = pd.DataFrame(np.array([total_irr_actual,total_irr_ET_R0,total_irr_SM_R0, total_irr_ET_R1, total_irr_SM_R1]).T, columns=columns, index=siteNames)

	yield_pd['ET_R0_saved'] = 100*(1- yield_pd['IA_ET_R0']/yield_pd['IA'])
	yield_pd['SM_R0_saved'] = 100 * (1 - yield_pd['IA_SM_R0'] / yield_pd['IA'])
	yield_pd['ET_R1_saved'] = 100 * (yield_pd['IA_ET_R0'] - yield_pd['IA_ET_R1'] )/ yield_pd['IA']
	yield_pd['SM_R1_saved'] = 100 * (yield_pd['IA_SM_R0'] - yield_pd['IA_SM_R1']) / yield_pd['IA']
	plt.rc('font', size=18)
	bar_width=0.4
	colors = ['#337AE3', '#5E96E9', '#80ACEE']
	x = range(len(siteNames))
	# bottom = np.zeros(len(siteNames))
	# for i in ['ET',
	# 		  'SM'
	# 		  ]:
	# 	fig, ax = plt.subplots()
	#
	# 	labels = ['{:.1f} %'.format(i) for i in yield_pd[i+'_R0_saved'].to_numpy()]
	# 	ax.bar(x,yield_pd['IA_'+i+'_R0'],bar_width, tick_label=siteNames, label="IA_"+i+"_R0", color=colors[0])
	# 	p = ax.bar(x,yield_pd['IA']-yield_pd['IA_'+i+'_R0'],bar_width, bottom = yield_pd['IA_'+i+'_R0'],
	# 	tick_label=siteNames, label='Irrigation Amount (IA)', color=colors[1])
	# 	ax.bar_label(p, labels, label_type='center')
	# 	plt.legend()
	# 	plt.gcf().set_size_inches(20, 10)
	# 	plt.tight_layout()
	# 	plt.savefig(r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m/irrigation_bar_'+i+'_R0.png', dpi=100)

	for i in ['ET',
			  'SM'
			  ]:
		fig, ax = plt.subplots()
		labels = ['{:.1f} %'.format(i) for i in yield_pd[i+'_R0_saved'].to_numpy()]
		labels2 = ['{:.1f} %'.format(i) for i in yield_pd[i+'_R1_saved'].to_numpy()]
		ax.bar(x,yield_pd['IA_'+i+'_R1'],bar_width, tick_label=siteNames, label="IA_"+i+"_R1", color=colors[0])
		p = ax.bar(x,yield_pd['IA_'+i+'_R0']-yield_pd['IA_'+i+'_R1'],bar_width,
			   bottom = yield_pd['IA_'+i+'_R1'],tick_label=siteNames, label="IA_"+i+"_R0", color=colors[1])
		ax.bar_label(p, labels2, label_type='center')

		p2 = ax.bar(x, yield_pd['IA'] - yield_pd['IA_' + i + '_R0'], bar_width,
				   bottom=yield_pd['IA_' + i + '_R0'], tick_label=siteNames, label='Irrigation Amount (IA)', color=colors[2])
		ax.bar_label(p2, labels, label_type='center')

		plt.legend()
		plt.gcf().set_size_inches(20, 10)
		plt.tight_layout()
		plt.savefig(r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m/irrigation_bar_'+i+'_R1.png', dpi=100)
	plt.show()


def plotLoss(resultPath):
	siteNames = ['EAST_2019', 'HOME_2019', 'KELLY_2019', 'LINKS_2019', 'NORTH_2020', 'HOME_2020', 'KELLY_2020',
				 'JOHNSON_2020']

	site = 'KELLY_2020'
	# lossFile = '/run-DQL_R1_H0-1677971062-tag-loss.csv'
	# rewardFile = '/run-DQL_R1_H0-1677971062-tag-reward_avg.csv'
	# returnFile = '/run-DQL_R1_H0-1677971062-tag-net_return_avg.csv'

	site = 'KELLY_2019'
	lossFile = '/run-UNL_KELLY_2019-DQL_R1_H0_a0.15-1678732664-tag-loss.csv'
	rewardFile = '/run-UNL_KELLY_2019-DQL_R1_H0_a0.15-1678732664-tag-reward_avg.csv'
	# returnFile = '/run-DQL_R1_H0-1677971062-tag-net_return_avg.csv'

	site = 'HOME_2019'
	lossFile = '/run-UNL_HOME_2019-DQL_R1_H0_a0.15-1678829369-tag-loss.csv'
	rewardFile = '/run-UNL_HOME_2019-DQL_R1_H0_a0.15-1678829369-tag-reward_avg.csv'

	loss_pd = pd.read_csv(resultPath+'/'+site.replace('_','')+lossFile, index_col='Step')
	reward_pd = pd.read_csv(resultPath+'/'+site.replace('_','')+rewardFile, index_col='Step')
	# return_pd = pd.read_csv(resultPath+'/'+site.replace('_','')+returnFile, index_col='Step')

	plt.rc('font', size=12)
	fig, axs = plt.subplots()
	l1 = loss_pd['Value'].iloc[20:].plot(ax=axs, ylabel='Loss',label='Loss', xlabel='Episode')
	ax2 = axs.twinx()
	l2 = reward_pd['Value'].iloc[5:].plot(ax=ax2, ylabel='Mean Reward',label='Mean Reward', color='C1', xlabel='Episode')
	# return_pd['Value'].plot(ax=axs[2])
	# ax2.set_yscale('log', base=2)

	# axs[2].set_ylabel('Return')
	fig.legend( bbox_to_anchor=(1,0.5), bbox_transform=axs.transAxes)
	plt.gcf().set_size_inches(10, 5)
	plt.tight_layout()
	plt.savefig(r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m'+'/'+site.replace('_','')+'/loss_reward_curve_DQL_R1.png', dpi=100)
	plt.show()

def plot3d():
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	xdata = [13.525,13.45,13.3,13.3,13.4,14.865,14.725,13.6,13.6,13.7]
	ydata = [131,122.5,130.3,121.8,206.8,243,237.5,226.7,222.5,278.4]
	zdata = [3477.4,3470.0,3405.52,3419.12,3291.12,3663.4,3633.0,3345.28,3352.0,3220.56]
	for i,marker in enumerate(['o','v','s','d']):
		ax.scatter3D(xdata[i], ydata[i], zdata[i], marker=marker, label=marker)

	plt.show()


def longestPalindrome(s):
	"""
    :type s: str
    :rtype: str
    """
	r = ''
	if len(s) == 1:
		return s
	for i in range(1, len(s) + 1):
		for j in range(len(s)):
			temp = s[j:j + i]
			print(temp)
			if temp == temp[::-1]:
				r = temp
				break
	return r

if __name__ == '__main__':
	# #main()
	resultPath = r'C:\Users\nuds\Dropbox\singlepoint\result\rootzone1m'
	# site = 'KELLY_2020'
	site = 'HOME_2019'
	# for i in ['threshold_ET.csv','threshold.csv' ]:
	#
	# 	plotThreshold(resultPath, i)
	# plotNetReturn()
	# plotAquaCropSM()
	# plotYield()
	# plotLoss(resultPath)
	# plot3d()
	longestPalindrome('babad')