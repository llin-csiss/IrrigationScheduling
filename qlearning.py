import numpy as np
import pandas as pd
import datetime as dt
import math
import random
import time
import subprocess
from subprocess import DEVNULL
from yieldEstimate import yield_run, getSoilProp
from irriTable import readVariable, getTexture, getWP, getAWC
from aquacrop.utils import prepare_weather, get_filepath
import json
from os.path import exists
from tqdm import tqdm
import os
import sys, getopt

g = np.random.Generator(np.random.PCG64())


class Env:
	def __init__(self,site_para, UPDATE_HRLDAS=False, USE_FORECASTED_RAINFALL=True):

		self.x = site_para['x']
		self.y = site_para['y']
		self.lat = site_para['lat']
		self.lon = site_para['lon']
		self.site_name = site_para['site_name']
		self.planting_date = site_para['planting_date']
		self.target_date = site_para['target_date']
		self.ETorSM = '{}_R{}_H{}'.format(site_para['ETorSM'], 1 if USE_FORECASTED_RAINFALL else 0,
												 1 if UPDATE_HRLDAS else 0)
		self.nc_file = str(site_para['planting_date']) + '.LDASOUT_DOMAIN1'
		self.start = dt.datetime.strptime(str(site_para['planting_date']), "%Y%m%d%H")
		self.end = dt.datetime.strptime(str(site_para['target_date']), "%Y%m%d%H")

		self.filePath = r"/home/hzhao/single-point/newoutput/" + str(self.x) + "/" + str(self.y) + "/"
		self.weather_path =  './input/dailyClimate.txt'  # climate data for UNL sites
		self.season_length = (self.end - self.start).days
		self.soil_info = getTexture(self.lat, self.lon)
		self.ksat_mean = 54.32 * 86.4  # getSoilProp(lat, lon, 'mean_ksat') * 86.4
		self.thwp = getWP(self.lat, self.lon) / 100
		self.thfc = self.thwp + getAWC(self.lat, self.lon)
		self.STATE_SIZE = 26 * self.season_length
		self.ACTION_SIZE = 42  # {0, 5, 5.5, 6, 6.5, …, 25}
		self.UPDATE_HRLDAS=UPDATE_HRLDAS
		self.USE_FORECASTED_RAINFALL = USE_FORECASTED_RAINFALL
		self.PUNISHMENT = -1000   # punishment when irrigating continuously

		self.corn_price = 280 # USD/ton
		self.amount_cost = 1.6 # USD/mm/ha
		self.depreciation_cost = 10 # USD/irrigation
		self.sm=pd.DataFrame()
		self.weather_data = prepare_weather(self.weather_path)

		self.state = self.reset()


	def sample(self):
		# select an action randomly
		return g.choice(self.ACTION_SIZE, 1)[0]

	def reset(self):
		#self.run_model(self.irrigation_data)
		self.action_record = []
		self.irrigation_data = "\'[]\'"
		self.steps = 0
		self.irrigation_amount=0
		self.total_irrigation_amount = 0 #total amount between plant and target date
		self.total_irrigated = 0 # total amount between plant and maturity date, effective irrigation
		self.irrigation_times = 0
		self.yield_est = 0
		if self.UPDATE_HRLDAS:
			self.sm = self.run_hrldas_model(self.irrigation_data)
		else:
			if self.sm.empty:self.update_sm()
		self.run_yield_estimate(False)
		current_sm = self.sm.iloc[self.steps,-1]
		max_sm = self.sm['SM'].max()
		min_sm = self.sm['SM'].min()
		inter = (max_sm-min_sm)/5
		fr = np.multiply(self.sm['RAIN'][self.steps + 1:self.steps + 6].values,
										  [0.9, 0.9 ** 2, 0.9 ** 3, 0.9 ** 4,
										   0.9 ** 5]).sum() if self.USE_FORECASTED_RAINFALL else 0
		rainfall_level = 0 if fr<5 else 1 if fr<10 else 2 if fr<15 else 3 if fr<20 else 4
		return math.floor((current_sm-min_sm)/inter) + 5 * rainfall_level

	def update_sm(self):
		ETPath = self.filePath + self.site_name + '_' + self.nc_file + '_ET.csv'
		if not exists(ETPath):
			self.sm = self.run_hrldas_model(self.irrigation_data,True)
			return
		et = pd.read_csv(ETPath, index_col=0)
		et.index = et.index.map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
		et = et.astype({"SM": float})
		self.sm = et

	def run_yield_estimate(self, write_to_file):
		irr_record = json.loads(self.irrigation_data.replace('\'',''))
		#para_kelly_2020['irr_record'] = irr_record
		y = yield_run(self.thwp, self.thfc, self.planting_date, self.target_date, irr_record, self.filePath, self.site_name, self.ETorSM, self.ksat_mean,
				  self.soil_info, self.weather_data, write_to_file)
		ye = y.loc[y['label'] == 'recommended schedule_' + self.ETorSM]
		self.yield_est = ye.iloc[0,0]
		self.total_irrigated = ye.iloc[0,1]
		#yield_est(**para_kelly_2020)

	def run_hrldas_model(self,irrigation_data,write_to_file=False):
		#start_string =start.strftime("%Y%m%d%H")
		nc_file = self.planting_date + '.LDASOUT_DOMAIN1'
		cmd = "./run_hrldas.sh " + str(self.x) + " " + str(self.y) + " " + self.planting_date + " " + self.target_date + " " + str(irrigation_data) + " " + str(
			self.lon) + " " + str(self.lat)
		print(cmd)
		time_start = time.perf_counter()
		p = subprocess.Popen(cmd, shell=True)
		p.wait()
		time_end = time.perf_counter()
		print('time cost: ', time_end - time_start, 's')
		sm = readVariable(nc_file, self.start, self.end, self.filePath, self.site_name, write_to_file)
		#os.system("rm " + self.filePath+nc_file)
		return sm
		#self.run_yield_estimate()

	def step(self, action):
		if action:
			self.update_irrigation_data(action)
			if self.UPDATE_HRLDAS:
				#start = self.start + dt.timedelta(days=self.steps)
				self.sm.update(self.run_hrldas_model(self.irrigation_data))
			reward = self.give_reward()
			next_state = 25+(self.steps+1)*26
		else:
			reward = 0
			next_state = self.get_next_state()
		self.update_action_record(action)
		self.steps += 1
		self.state = next_state
		done = True if self.steps>=self.season_length-1 else False
		return next_state, reward, done

	def get_next_state(self):
		# whether irrigated in last three days
		next_steps = self.steps+1
		if (len(self.action_record)>=1 and self.action_record[-1]):
			next_state = 25+next_steps*26
		else:
			step_date = self.start + dt.timedelta(days=next_steps)
			next_sm = self.sm.at[step_date,'SM']
			max_sm = self.sm['SM'].max()
			min_sm = self.sm['SM'].min()
			inter = (max_sm-min_sm)/5
			# Forecasted rainfall volume in next 5 days
			fr = np.multiply(self.sm['RAIN'][next_steps + 1:next_steps + 6].values,
											  [0.9, 0.9 ** 2, 0.9 ** 3, 0.9 ** 4, 0.9 ** 5]).sum() \
				if next_steps + 6 < self.season_length and self.USE_FORECASTED_RAINFALL else 0
			rainfall_level = 0 if fr < 5 else 1 if fr < 10 else 2 if fr < 15 else 3 if fr < 20 else 4
			next_state = math.floor((next_sm-min_sm)/inter) + 5 * rainfall_level + (self.steps+1)*26
		return next_state

	def give_reward(self):
		previous_yield = self.yield_est
		self.run_yield_estimate(False)
		current_yield = self.yield_est
		reward = (current_yield - previous_yield) * self.corn_price - self.irrigation_amount * self.amount_cost - (
			self.depreciation_cost if self.irrigation_amount else 0)
		if self.state%26 == 25:
			reward +=self.PUNISHMENT
		# 	print(self.action_record)
		return reward


	def update_action_record(self,action):
		self.action_record.append(action)

	def update_irrigation_data(self,action):
		action_list = [0] + [i / 2 for i in range(10, 51)]
		irrigation_amount = action_list[action]
		t = self.start + dt.timedelta(days=self.steps)
		t_str = t.strftime("%Y-%m-%d")+'T00:00:00.000Z'
		new_irr = '{\"start\":\"'+t_str+'\",\"end\":\"'+t_str+'\",\"volume\":\"'+str(irrigation_amount)+'\",\"changed\":\"true\"'+'}'
		temp = self.irrigation_data.replace("true","false")
		if len(json.loads(self.irrigation_data.replace('\'','')))>0:
			temp = temp[:-2]+','+new_irr+']\''
		else:
			temp = temp[:-2] + new_irr + ']\''
		self.irrigation_data = temp
		self.irrigation_amount = irrigation_amount
		self.total_irrigation_amount += irrigation_amount
		self.irrigation_times +=1

class Agent:
	def __init__(self, STATE_SIZE, ACTION_SIZE, alpha = 0.25, gamma = 0.999, epsilon = 1, epsilon_decay = 0.995):
		self.action_history=[]
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = 0.1
		self.q_table = np.zeros([STATE_SIZE, ACTION_SIZE])
		self.action_size = ACTION_SIZE

	def choose_action(self, state):
		if random.uniform(0, 1) < self.epsilon:
			return g.choice(self.action_size, 1)[0]
		else:
			return np.argmax(self.q_table[state])

	def init_q_table_from_file(self, q_table_file):
		if exists(q_table_file):
			print("Existing q table found!")
			q_table = pd.read_csv(q_table_file, sep=',', header=None)
			self.q_table = q_table.values

	def update_q_table(self, action, state, next_state, reward):
		# update value in q_table
		old_value = self.q_table[state, action]
		next_max = np.max(self.q_table[next_state])

		new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
		self.q_table[state, action] = new_value

	def update_action_history(self,action, reward):
		# update action history
		self.action_history.append((action, reward))

	def update_epsilon(self):
		if self.epsilon>self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		else:
			self.epsilon = self.epsilon_min

def q_learning(site_para, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter, USE_FORECASTED_RAINFALL):
	# site_para['ETorSM'] = '{}_{}_R{}_H{}'.format(site_para['ETorSM'], alpha, 1 if USE_FORECASTED_RAINFALL else 0,
	# 											 1 if UPDATE_HRLDAS else 0)
	SAVE_MODEL_EVERY = 50
	# for plotting metrics
	net_returns = []
	all_penalties = []
	action_history = []

	env = Env(site_para, UPDATE_HRLDAS, USE_FORECASTED_RAINFALL)
	agent = Agent(env.STATE_SIZE, env.ACTION_SIZE,alpha, gamma, epsilon, epsilon_decay)

	# if UPDATE_HRLDAS:
	# 	q_table_file = env.filePath+env.site_name+"_"+env.ETorSM+"_q-table.csv"
	# 	agent.init_q_table_from_file(q_table_file)

	for i in tqdm(range(1, maxiter + 1), ascii=True, unit='episodes'):
		state = env.reset()
		done = False
		while not done:
			action = agent.choose_action(state)
			next_state, reward, done = env.step(action)
			agent.update_action_history(action, reward)
			agent.update_q_table(action, state, next_state, reward)
			# print(f"q-table:{q_table}")
			# print(f"action:{action}, state:{state}, reward:{reward}, next_max:{next_max}, next state:{next_state}")
			state = next_state
			# input("Press Enter to continue...")
		net_return = env.yield_est*env.corn_price - env.total_irrigation_amount*env.amount_cost - env.irrigation_times*env.depreciation_cost
		net_returns.append(net_return)
		agent.update_epsilon()
		#epsilon = epsilon * epsilon_decay #epsilon greedy policy
		if UPDATE_HRLDAS:
			cmd = "./deleteIrrigatedDays.sh " + str(env.x) + " " +str(env.y)+\
				   " " + env.irrigation_data
			p = subprocess.Popen(cmd, shell=True)
			p.wait()

		# action_history = agent.action_history
		agent.action_history = []
		# print(f"q-table:{q_table}")
		# print(f"action history:{action_history}")
		# print(f"net return:{net_return},irrigation amount:{env.total_irrigation_amount},yield:{env.yield_est}, irrigation in season:{env.total_irrigated}")
		# print(f"Episode:{i}, epsilon:{agent.epsilon}")
		# input("Press Enter to continue...")
		if not i % SAVE_MODEL_EVERY:
			np.savetxt(env.filePath + site_para['site_name'] + '_'+env.ETorSM+f'_a{alpha}'+'_q-table.csv', agent.q_table, delimiter=",", fmt='%s')
			# env.run_yield_estimate(True)
			np.savetxt(env.filePath+site_para['site_name']+'_'+env.ETorSM+f'_a{alpha}'+'_net_return.csv', net_returns, delimiter=",",fmt='%s')
			# np.savetxt(env.filePath+site_para['site_name']+'_'+env.ETorSM+f'_a{alpha}'+'_action_history.csv', action_history, delimiter=",",fmt='%s')


def test(site_para,q_table_file):
	# site_para['ETorSM'] = '{}_{}_TEST'.format(site_para['ETorSM'], model_name, )
	if exists(q_table_file):

		q_table = pd.read_csv(q_table_file, sep=',', header=None)
		q_table = q_table.values
		env2 = Env(site_para)
		env2.ETorSM = q_table_file.split('/')[-1].split('q-table')[0]
		action_history = []
		done = False
		while not done:
			if env2.state> len(q_table)-1:
				break
			action = np.argmax(q_table[env2.state])
			# print(action)
			next_state, reward, done = env2.step(action)
			action_history.append((action, reward))

		net_return = env2.yield_est * env2.corn_price - env2.total_irrigation_amount * env2.amount_cost - env2.irrigation_times * env2.depreciation_cost
		env2.run_yield_estimate(True)
		print('net return:{}'.format(net_return))
		np.savetxt(env2.filePath + site_para['site_name'] + '_' + env2.ETorSM + '_test_net_return.csv',
				   [net_return], delimiter=",", fmt='%s')
		np.savetxt(env2.filePath + site_para['site_name'] + '_' + env2.ETorSM + '_test_action_history.csv',
				   action_history, delimiter=",", fmt='%s')
	else:
		print("q table not found!")

def main(argv):
	train_flag=0
	validate_flag=0
	year='2019'
	try:
		opts, args = getopt.getopt(argv,'ht:v:y:',['train=','validate=','year='])
	except getopt.GetoptError:
		print('qlearning.py -t <0 or 1 tag train> '
			  '-v <0 or 1 tag for validation> -y <year, 2019 or 2020>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('qlearning.py -t <0 or 1 tag train> '
				  '-v <0 or 1 tag for validation> -y <year, 2019 or 2020>')
			sys.exit()
		elif opt in ('-t','--train'):
			train_flag = int(arg)
		elif opt in ('-v','--validate'):
			validate_flag = int(arg)
		elif opt in ('-y','--year'):
			year = arg
	para_kelly2020 = {'lon': -98.2034475, 'lat': 41.9480136, 'wp': 0, 'awc': 0, 'planting_date': '2020050100',
					  'target_date': '2020093023', 'irr_record': [],
					  'x': 986, 'y': 448, 'site_name': "UNL_KELLY_2020", 'dep_thld': '[22,35,52,15]', 'ETorSM': "QL"
					  }
	para_kelly2019 = {'lon': -98.2034475, 'lat': 41.9480136, 'wp': 0, 'awc': 0, 'planting_date': '2019042500',
					  'target_date': '2019093023', 'irr_record': [],
					  'x': 986, 'y': 448, 'site_name': "UNL_KELLY_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "QL"}
	para_home2020 = {'lon': -98.1970, 'lat': 41.94144, 'wp': 0, 'awc': 0, 'planting_date': '2020050100',
					 'target_date': '2020093023', 'irr_record': [],
					 'x': 987, 'y': 447, 'site_name': "UNL_HOME_2020", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "QL"}

	para_links2019 = {'lon': -98.21542, 'lat': 41.95572, 'wp': 0, 'awc': 0, 'planting_date': '2019042400',
					  'target_date': '2019093023', 'irr_record': [],
					  'x': 984, 'y': 450, 'site_name': "UNL_LINKS_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "QL"}

	para_home2019 = {'lon': -98.1970, 'lat': 41.94144, 'wp': 0, 'awc': 0, 'planting_date': '2019050400',
					 'target_date': '2019093023', 'irr_record': [],
					 'x': 987, 'y': 447, 'site_name': "UNL_HOME_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "QL"}

	para_east2019 = {'lon': -98.18372, 'lat': 41.9406, 'wp': 0, 'awc': 0, 'planting_date': '2019050200',
					 'target_date': '2019093023', 'irr_record': [],
					 'x': 989, 'y': 447, 'site_name': "UNL_EAST_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "QL"}
	para_johnson2020 = {'lon': -98.21608, 'lat': 42.02841, 'wp': 0, 'awc': 0, 'planting_date': '2020042500',
						'target_date': '2020093023', 'irr_record': [],
						'x': 984, 'y': 466, 'site_name': "UNL_JOHNSON_2020", 'dep_thld': [50, 50, 50, 50],
						'ETorSM': "QL"}
	para_north2020 = {'lon': -98.19706, 'lat': 41.94675, 'wp': 0, 'awc': 0, 'planting_date': '2020050800',
					  'target_date': '2020093023', 'irr_record': [],
					  'x': 987, 'y': 448, 'site_name': "UNL_NORTH_2020", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "QL"}


	para_dict = {
		# "UNL_HOME_2019": para_home2019,
		# "UNL_EAST_2019": para_east2019,
		# "UNL_KELLY_2019": para_kelly2019,
		# "UNL_LINKS_2019": para_links2019,
		# "UNL_JOHNSON_2020": para_johnson2020,
		"UNL_KELLY_2020": para_kelly2020,
		# "UNL_HOME_2020": para_home2020,
		# "UNL_NORTH_2020": para_north2020,
	}
	crop_type = "Corn"
	for key, value in para_dict.items():
		print(f"######################################{key}")
		filePath = r"/home/hzhao/single-point/newoutput/" + str(value['x']) + "/" + str(value['y']) + "/"
		######################################################################################################
		# Hyperparameters
		UPDATE_HRLDAS = False
		USE_FORECASTED_RAINFALL = False
		alpha = 0.15  # learning rate
		gamma = 0.9999  # discount factor
		epsilon_init = 1
		epsilon_decay = 0.985
		epsilon = epsilon_init
		maxiter = 650


		# model_name = '{}_R{}_H{}_a{}'.format(value['ETorSM'], 1 if USE_FORECASTED_RAINFALL else 0,
		# 										 1 if UPDATE_HRLDAS else 0, alpha)
		# # Train
		# if train_flag:
		# 	if not(exists(filePath+value['site_name'] + '_'+model_name+'_q-table.csv')):
		# 		q_learning(value, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter, USE_FORECASTED_RAINFALL)
		# 	else:
		# 		print('training for this site is completed!')
        #
        #
		# # Validate
		# # model_name = '{}_R{}_H{}'.format(value['ETorSM'], 1 if USE_FORECASTED_RAINFALL else 0,
		# # 1 if UPDATE_HRLDAS else 0)
		# #
		# if validate_flag:
		# 	q_table_file = filePath + value['site_name'] + "_" + model_name + "_q-table.csv"
		# 	test(value,q_table_file)

		# Train
		USE_FORECASTED_RAINFALL = True
		UPDATE_HRLDAS = True
		model_name ='{}_R{}_H{}_a{}'.format(value['ETorSM'],1 if USE_FORECASTED_RAINFALL else 0,
											1 if UPDATE_HRLDAS else 0, alpha)
		if train_flag:
			if not (exists(filePath + value['site_name'] + '_' + model_name + '_q-table.csv')):
				q_learning(value, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter, USE_FORECASTED_RAINFALL)
			else:
				print('training for this site is completed!')

		# Validate
		# model_name = '{}_R{}_H{}'.format(value['ETorSM'], 1 if USE_FORECASTED_RAINFALL else 0,
		# 									 1 if UPDATE_HRLDAS else 0)
		if validate_flag:
			q_table_file = filePath + value['site_name'] + "_" + model_name + "_q-table.csv"

			# test(value, q_table_file)
			for a, b in para_dict.items():
				test(b, q_table_file)
	# Test
	# model_name =  "QL"
	# q_table_file = filePath + para_kelly_2020['site_name'] + "_" + model_name + "_q-table.csv"
	# q_table=[]
	# if exists(q_table_file):
	# 	print("Existing q table found!")
	# 	q_table = pd.read_csv(q_table_file, sep=',', header=None)
	# 	q_table = q_table.values
	#
	# print(len(q_table))
	# test(para_kelly_2019,q_table,model_name)
	# test(para_home_2020,q_table,model_name)
	#######################################################################################################
	# Hyperparameters
	# UPDATE_HRLDAS = True
	# alpha = 0.75  # learning rate
	# gamma = 0.9999  # discount factor
	# epsilon_init = 1
	# epsilon_decay = 0.985
	# epsilon = epsilon_init
	# maxiter = 650
	#
	# q_learning(para_kelly_2020, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter)


if __name__ == '__main__':
	main(sys.argv[1:])

