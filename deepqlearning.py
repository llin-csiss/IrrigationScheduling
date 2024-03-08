import numpy as np
import pandas as pd
import datetime as dt
import math
import random
import time
import subprocess
from yieldEstimate import yield_run, getSoilProp
from irriTable import readVariable, getTexture, getWP, getAWC
from aquacrop.utils import prepare_weather, get_filepath
import json
from os.path import exists

from collections import deque
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH=true']='true'
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K
from keras.callbacks import TensorBoard
import tensorflow as tf

from tqdm import tqdm

import sys, getopt

from tensorflow.python.client import device_lib

gpus = tf.config.list_physical_devices('GPU')
# if gpus:
# 	try:
# 		# Currently, memory growth needs to be the same across GPUs
# 		for gpu in gpus:
# 			tf.config.experimental.set_memory_growth(gpu, True)
# 		logical_gpus = tf.config.list_logical_devices('GPU')
# 		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
# 	except RuntimeError as e:
# 		# Memory growth must be set before GPUs have been initialized
# 		print(e)

# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=100, \
#                         inter_op_parallelism_threads=10, \
#                         allow_soft_placement=True, \
#                         device_count = {'GPU': 4})
# session = tf.compat.v1.Session(config=config)
# K.set_session(session)

g = np.random.Generator(np.random.PCG64())

class ModifiedTensorBoard(TensorBoard):

	def __init__(self,name, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.create_file_writer(self.log_dir)
		self._log_write_dir =  os.path.join(self.log_dir, name)

	def set_model(self, model):
		self.model = model

		self._train_dir = os.path.join(self._log_write_dir, 'train')
		self._train_step = self.model._train_counter

		self._val_dir = os.path.join(self._log_write_dir, 'validation')
		self._val_step = self.model._test_counter

		self._should_write_train_graph = False

	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	def on_batch_end(self, batch, logs=None):
		pass

	def on_train_end(self, _):
		pass

	# Custom method for saving own metrics
	# Creates writer, writes custom metrics and closes writer
	def update_stats(self, **stats):
		with self.writer.as_default():
			for key, value in stats.items():
				tf.summary.scalar(key, value, step = self.step)
				self.step += 1
				self.writer.flush()
class Env:
	STATE_SIZE = 7  # dimension of the state ### 5 * self.season_length
	ACTION_SIZE = 42  # {0, 5, 5.5, 6, 6.5, …, 25}
	PUNISHMENT = -1000  # punishment when irrigating continuously

	corn_price = 280  # USD/ton
	amount_cost = 1.6  # USD/mm/ha # 32.96 usd per acre-foot
	depreciation_cost = 10  # USD/irrigation


	weather_path = './input/dailyClimate.txt'  # climate data for UNL sites


	def __init__(self,site_para, UPDATE_HRLDAS=False, USE_FORECASTED_RAINFALL=False):

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
		self.season_length = (self.end - self.start).days
		self.soil_info = getTexture(self.lat, self.lon)
		self.ksat_mean = 54.32 * 86.4  # getSoilProp(lat, lon, 'mean_ksat') * 86.4
		self.thwp = getWP(self.lat, self.lon) / 100
		self.thfc = self.thwp + getAWC(self.lat, self.lon)
		self.UPDATE_HRLDAS=UPDATE_HRLDAS
		self.USE_FORECASTED_RAINFALL = USE_FORECASTED_RAINFALL
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
		crop_stage = 0 if self.steps<=25 else 1 if self.steps<=60 else 2 if self.steps<=100 else 3
		precip = self.sm.iloc[self.steps]['RAIN']
		ET = self.sm.iloc[self.steps]['ET']
		ET = float(ET) if ET != '--' else 0
		forecasted_rainfall = np.multiply(self.sm['RAIN'][self.steps+1:self.steps+6].values,
										  [0.9,0.9**2, 0.9**3,0.9**4,0.9**5]).sum() if self.USE_FORECASTED_RAINFALL else 0
		irrigated = 0 # whether irrigated in last three days

		state = [self.steps, crop_stage, precip, ET, current_sm, forecasted_rainfall, irrigated ]
		return state

	def update_sm(self):
		ETPath = self.filePath + self.site_name + '_' + self.nc_file + '_ET.csv'
		if not exists(ETPath):
			self.sm = self.run_hrldas_model(self.irrigation_data, self.start,True)
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
		ye = y.loc[y['label'] == 'recommended schedule_' + self.ETorSM ]
		self.yield_est = ye.iloc[0,0]
		self.total_irrigated = ye.iloc[0,1]
		#yield_est(**para_kelly_2020)

	def run_hrldas_model(self,irrigation_data,write_to_file=False): # update sm and et in the days following irrigation
		# start_string = start.strftime("%Y%m%d%H")
		nc_file = self.planting_date + '.LDASOUT_DOMAIN1'
		cmd = "./run_hrldas.sh " + str(self.x) + " " + str(self.y) + " " + self.planting_date\
			  + " " + str(self.target_date) + " " + str(irrigation_data) + " " + str(
			self.lon) + " " + str(self.lat)
		print(cmd)
		time_start = time.perf_counter()
		p = subprocess.Popen(cmd, shell=True)
		p.wait()
		time_end = time.perf_counter()
		print('time cost: ', time_end - time_start, 's')
		sm = readVariable(nc_file, self.start, self.end, self.filePath, self.site_name, write_to_file)
		# os.system("rm " + self.filePath + nc_file)
		return sm

	def step(self, action):
		if action:
			self.update_irrigation_data(action)
			if self.UPDATE_HRLDAS:
				self.sm.update(self.run_hrldas_model(self.irrigation_data))
			reward = self.give_reward()
			irrigated = 1
		else:
			reward = 0
			if (len(self.action_record) >= 1 and self.action_record[-1]):
				irrigated = 1
			else: irrigated = 0
		next_state = self.get_next_state(irrigated)
		self.update_action_record(action)
		self.steps += 1
		self.state = next_state
		terminated = True if self.steps>=self.season_length-1 else False
		return next_state, reward, terminated

	def get_next_state(self, irrigated):
		#self.update_sm()
		next_steps = self.steps+1
		next_sm = self.sm.iloc[next_steps,-1]
		crop_stage = 0 if next_steps <= 25 else 1 if next_steps <= 60 else 2 if next_steps <= 100 else 3
		precip = self.sm.iloc[next_steps]['RAIN']
		ET = self.sm.iloc[next_steps]['ET']
		ET = float(ET) if ET != '--' else 0
		forecasted_rainfall = np.multiply(self.sm['RAIN'][next_steps + 1:next_steps + 6].values,
							[0.9, 0.9 ** 2, 0.9 ** 3, 0.9 ** 4, 0.9 ** 5]).sum() \
			if next_steps+6<self.season_length and self.USE_FORECASTED_RAINFALL else 0

		next_state = [next_steps, crop_stage, precip, ET, next_sm, forecasted_rainfall, irrigated]
		return next_state

	def give_reward(self):
		previous_yield = self.yield_est
		self.run_yield_estimate(False)
		current_yield = self.yield_est
		reward = (current_yield - previous_yield) * self.corn_price - self.irrigation_amount * self.amount_cost - (
			self.depreciation_cost if self.irrigation_amount else 0)
		if self.state[-1]:
			reward +=self.PUNISHMENT
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
	def __init__(self, env, optimizer, alpha = 0.25, gamma = 0.999, epsilon = 1, epsilon_decay = 0.995 ):
		self.action_history=[]
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = 0#0.001
		#self.q_table = np.zeros([STATE_SIZE, ACTION_SIZE])
		self.action_size = env.ACTION_SIZE
		self.state_dimension = env.STATE_SIZE

		self._optimizer = optimizer
		# An array with last n steps for training
		self.experience_replay = deque(maxlen=10000)

		# build networks
		self.q_network = self._build_compile_model()
		self.target_network = self._build_compile_model()
		print(self.q_network.summary())
		self.align_target_model()

		# Custom tensorboard object
		self.tensorboard = ModifiedTensorBoard(env.ETorSM, log_dir="{}/logs/{}-{}_a{}-{}".format(env.filePath, env.site_name, env.ETorSM, self.alpha, int(time.time())))

	def store(self, state, action, reward, next_state, terminated):
		self.experience_replay.append((state, action, reward, next_state, terminated))

	def _build_compile_model(self):
		# mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
		# # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
		# with mirrored_strategy.scope():
		model = Sequential()
		#model.add(Embedding)
		model.add(Dense(24, input_dim=self.state_dimension, activation='relu'))
		model.add(Dense(24,activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=self._optimizer)
		return model

	def align_target_model(self):
		self.target_network.set_weights(self.q_network.get_weights())

	def act(self, state):
		if random.uniform(0, 1) < self.epsilon:
			return g.choice(self.action_size, 1)[0]
		else:
			q_values = self.q_network.predict(np.array(state).reshape(-1,self.state_dimension,1), verbose=0)
			return np.argmax(q_values[0])

	# Trains main network every step during episode
	def retrain(self,batch_size,terminal_state):
		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.experience_replay, batch_size)
		# Get current states from minibatch, then query NN model for Q values
		current_states = np.array([transition[0] for transition in minibatch])
		current_qs_list = self.q_network.predict(current_states.reshape(-1,self.state_dimension,1), verbose=0)

		# Get future states from minibatch, then query NN model for Q values
		# When using target network, query it, otherwise main network should be queried
		new_current_states = np.array([transition[3] for transition in minibatch])
		future_qs_list = self.target_network.predict(new_current_states.reshape(-1,self.state_dimension,1), verbose=0)

		x=[]
		y=[]

		for index, (current_state, action, reward, new_current_state, terminated) in enumerate(minibatch):

			# If not a terminal state, get new q from future states, otherwise set it to 0
			# almost like with Q Learning, but we use just part of equation here
			if terminated:
				new_q = reward
			else:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + self.gamma*max_future_q

			# Update Q value for given state
			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			# And append to our training data
			x.append(current_state)
			y.append(current_qs)
		self.q_network.fit(np.array(x).reshape(-1, self.state_dimension, 1), np.array(y), batch_size=batch_size, verbose=0,
						   shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

	def update_action_history(self,action, reward):
		# update action history
		self.action_history.append((action, reward))

	def update_epsilon(self):
		if self.epsilon>self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		else:
			self.epsilon = self.epsilon_min

def save_model_and_weights(agent, model_name, path, ):
	agent.q_network.save(f'{path}/models/{model_name}')
	best_weights = agent.q_network.get_weights()
	return best_weights

def dq_learning(site_para, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter, USE_FORECASTED_RAINFALL=True, init_weights =None, init_iter=1):
	# site_para['ETorSM'] = '{}_{}_R{}_H{}'.format(site_para['ETorSM'], alpha, 1 if USE_FORECASTED_RAINFALL else 0, 1 if UPDATE_HRLDAS else 0)
	SAVE_MODEL_EVERY = 50
	AGGREGATE_STATS_EVERY = 20
	# for plotting metrics
	net_returns = []
	ep_rewards = []
	weights = []

	optimizer = Adam(learning_rate=0.001)
	env = Env(site_para, UPDATE_HRLDAS, USE_FORECASTED_RAINFALL)
	agent = Agent(env,optimizer, alpha, gamma, epsilon, epsilon_decay)
	if init_weights: agent.q_network.set_weights(init_weights)
	batch_size = 150
	print('Start')
	for i in tqdm(range(init_iter,maxiter+1), ascii=True, unit='episodes'):
		agent.tensorboard.step = i
		episode_reward = 0
		state = env.reset()

		terminated = False
		while not terminated:
			#Run action
			action = agent.act(state)
			#Take action
			next_state, reward, terminated = env.step(action)
			episode_reward += reward
			agent.update_action_history(action, reward)
			agent.store(state, action, reward, next_state, terminated)

			state = next_state
			# Start training only if certain number of samples is already saved
			if len(agent.experience_replay)>batch_size:
				agent.retrain(batch_size,terminated)
		#update target network with weights of q network
		agent.align_target_model()
		net_return = env.yield_est*env.corn_price - env.total_irrigation_amount*env.amount_cost - env.irrigation_times*env.depreciation_cost
		net_returns.append(net_return)
		# Append episode reward to a list and log stats (every given number of episodes)
		ep_rewards.append(episode_reward)
		if not i % AGGREGATE_STATS_EVERY:
			average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
			min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
			max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
			average_net_return = sum(net_returns[-AGGREGATE_STATS_EVERY:]) / len(net_returns[-AGGREGATE_STATS_EVERY:])
			min_net_return = min(net_returns[-AGGREGATE_STATS_EVERY:])
			max_net_return = max(net_returns[-AGGREGATE_STATS_EVERY:])
			agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
										   net_return_avg=average_net_return, net_return_min=min_net_return,net_return_max=max_net_return,
										   epsilon=agent.epsilon)
		agent.update_epsilon() #epsilon greedy policy
		if UPDATE_HRLDAS:
			cmd = "./deleteIrrigatedDays.sh " + str(env.x) + " " +str(env.y)+\
				   " " + env.irrigation_data
			p = subprocess.Popen(cmd, shell=True)
			p.wait()

		action_history = agent.action_history
		agent.action_history = []
		# print(f"q-table:{q_table}")
		# print(f"action history:{action_history}")
		# print(f"net return:{net_return},irrigation amount:{env.total_irrigation_amount},yield:{env.yield_est}, irrigation in season:{env.total_irrigated}")
		# print(f"Episode:{i}, epsilon:{agent.epsilon}")
		# input("Press Enter to continue...")
		if not i % SAVE_MODEL_EVERY:
			weights = save_model_and_weights(agent,f"{site_para['site_name']}-{env.ETorSM}_a{agent.alpha}-iter{i}.model", env.filePath)
			# env.run_yield_estimate(True)
			np.savetxt(env.filePath+site_para['site_name']+'_'+env.ETorSM+f'_a{agent.alpha}_net_return.csv', net_returns, delimiter=",",fmt='%s')
			# np.savetxt(env.filePath+site_para['site_name']+'_'+env.ETorSM+'_action_history.csv', action_history, delimiter=",",fmt='%s')
	print("######################################Finished!")
	return weights

def test(site_para, model_file, UPDATE_HRLDAS, USE_FORECASTED_RAINFALL,alpha):

	model = load_model(model_file)
	env2 = Env(site_para, UPDATE_HRLDAS, USE_FORECASTED_RAINFALL)
	env2.ETorSM = model_file.split('/')[-1].split('.')[0]
	terminated = False
	action_history = []
	while not terminated:
		prd = model.predict(np.array(env2.state).reshape(-1,env2.STATE_SIZE,1), verbose=0)
		action =  np.argmax(prd[0])
		_, reward, terminated = env2.step(action)
		action_history.append((action, reward))
	net_return = env2.yield_est*env2.corn_price - env2.total_irrigation_amount*env2.amount_cost - env2.irrigation_times*env2.depreciation_cost
	env2.run_yield_estimate(True)
	print('net return:{}'.format(net_return))
	np.savetxt(env2.filePath + site_para['site_name'] + '_' + env2.ETorSM + f'_test_net_return.csv',
			   [net_return], delimiter=",", fmt='%s')
	np.savetxt(env2.filePath + site_para['site_name'] + '_' + env2.ETorSM + f'_test_action_history.csv',
			   action_history, delimiter=",", fmt='%s')


def main(argv):
	train_flag=0
	validate_flag=0
	year='2020'
	continue_flag = 0
	try:
		opts, args = getopt.getopt(argv,'ht:v:y:c:',['train=','validate=','year=', 'continue='])
	except getopt.GetoptError:
		print('deepqlearning.py -t <0 or 1 flag train> '
			  '-v <0 or 1 flag for validation> -y <year, 2019 or 2020> -c <0 or 1 flag for continue from last train>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('deepqlearning.py -t <0 or 1 flag train> '
				  '-v <0 or 1 flag for validation> -y <year, 2019 or 2020> -c <0 or 1 flag for continue from last train')
			sys.exit()
		elif opt in ('-t','--train'):
			train_flag = int(arg)
		elif opt in('-c','--continue'):
			continue_flag = int(arg)
		elif opt in ('-v','--validate'):
			validate_flag = int(arg)
		elif opt in ('-y','--year'):
			year = arg
	para_kelly2020 = {'lon': -98.2034475, 'lat': 41.9480136, 'wp': 0, 'awc': 0, 'planting_date': '2020050100',
					  'target_date': '2020093023', 'irr_record': [],
					  'x': 986, 'y': 448, 'site_name': "UNL_KELLY_2020", 'dep_thld': '[22,35,52,15]', 'ETorSM': "DQL"
					  }
	para_kelly2019 = {'lon': -98.2034475, 'lat': 41.9480136, 'wp': 0, 'awc': 0, 'planting_date': '2019042500',
					  'target_date': '2019093023', 'irr_record': [],
					  'x': 986, 'y': 448, 'site_name': "UNL_KELLY_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "DQL"}
	para_home2020 = {'lon': -98.1970, 'lat': 41.94144, 'wp': 0, 'awc': 0, 'planting_date': '2020050100',
					 'target_date': '2020093023', 'irr_record': [],
					 'x': 987, 'y': 447, 'site_name': "UNL_HOME_2020", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "DQL"}
	para_links2019 = {'lon': -98.21542, 'lat': 41.95572, 'wp': 0, 'awc': 0, 'planting_date': '2019042400',
					  'target_date': '2019093023', 'irr_record': [],
					  'x': 984, 'y': 450, 'site_name': "UNL_LINKS_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "DQL"}
	para_home2019 = {'lon': -98.1970, 'lat': 41.94144, 'wp': 0, 'awc': 0, 'planting_date': '2019050400',
					 'target_date': '2019093023', 'irr_record': [],
					 'x': 987, 'y': 447, 'site_name': "UNL_HOME_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "DQL"}
	para_east2019 = {'lon': -98.18372, 'lat': 41.9406, 'wp': 0, 'awc': 0, 'planting_date': '2019050200',
					 'target_date': '2019093023', 'irr_record': [],
					 'x': 989, 'y': 447, 'site_name': "UNL_EAST_2019", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "DQL"}
	para_johnson2020 = {'lon': -98.21608, 'lat': 42.02841, 'wp': 0, 'awc': 0, 'planting_date': '2020042500',
						'target_date': '2020093023', 'irr_record': [],
						'x': 984, 'y': 466, 'site_name': "UNL_JOHNSON_2020", 'dep_thld': [50, 50, 50, 50],
						'ETorSM': "DQL"}
	para_north2020 = {'lon': -98.19706, 'lat': 41.94675, 'wp': 0, 'awc': 0, 'planting_date': '2020050800',
					  'target_date': '2020093023', 'irr_record': [],
					  'x': 987, 'y': 448, 'site_name': "UNL_NORTH_2020", 'dep_thld': [50, 50, 50, 50], 'ETorSM': "DQL"}

	if year == '2019':
		para_dict = {
			"UNL_HOME_2019": para_home2019,
			"UNL_EAST_2019": para_east2019,
			"UNL_KELLY_2019": para_kelly2019,
			"UNL_LINKS_2019": para_links2019
					 }
	elif year == '2020':
		para_dict = {
			"UNL_JOHNSON_2020": para_johnson2020,
			"UNL_KELLY_2020": para_kelly2020,
			"UNL_HOME_2020": para_home2020,
			"UNL_NORTH_2020": para_north2020,
		}
	crop_type = "Corn"
	for key, value in para_dict.items():
		print(f"######################################{key}")
		###################################################################################################
		# step 1, train without HRLDAS update
		file_path = r"/home/hzhao/single-point/newoutput/" + str(value['x']) + "/" + str(value['y']) + "/"
		# Hyperparameters
		UPDATE_HRLDAS = False
		USE_FORECASTED_RAINFALL = False
		alpha = 0.15  # learning rate
		gamma = 0.9999  # discount factor
		epsilon_init = 1  # 0.0005225348357666069
		epsilon_decay = 0.985
		# epsilon_min = 0.001 # define in Agent class
		epsilon = epsilon_init
		maxiter = 1000

		# Train

		USE_FORECASTED_RAINFALL = True
		model_name = '{}-{}_R{}_H{}_a{}-iter{}.model'.format(value['site_name'], value['ETorSM'],
														 1 if USE_FORECASTED_RAINFALL else 0,
															 1 if UPDATE_HRLDAS else 0, alpha, maxiter)
		if train_flag:
			if not (exists(file_path+'models/'+model_name)):
				dq_learning(value, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter,
						USE_FORECASTED_RAINFALL)
			else:
				print('training for this site is completed!')

		if continue_flag:
			iters = [900,850,800,750,700,650,600,550,500]
			# epsilon = 0.01
			for last_iter in iters:
				epsilon = 0.985 ** last_iter # in case epsilon_min = 0
				model_name = '{}-{}_R{}_H{}_a{}-iter{}.model'.format(value['site_name'], value['ETorSM'],
																	 1 if USE_FORECASTED_RAINFALL else 0,
																	 1 if UPDATE_HRLDAS else 0, alpha, last_iter)
				if (exists(file_path+'models/'+model_name)):
					model_file = f'{file_path}/models/{model_name}'
					model = load_model(model_file)
					weights = model.get_weights()
					dq_learning(value, UPDATE_HRLDAS, alpha, gamma, epsilon, epsilon_decay, maxiter,
								USE_FORECASTED_RAINFALL, weights,last_iter+1)
					break

		# Validate
		if validate_flag:
			iters = [900,850,800,750,700,650,600,550,500]
			for last_iter in iters:
				model_name = '{}-{}_R{}_H{}_a{}-iter{}.model'.format(value['site_name'], value['ETorSM'],
																	 1 if USE_FORECASTED_RAINFALL else 0,
																	 1 if UPDATE_HRLDAS else 0, alpha, last_iter)
				model_file = f'{file_path}/models/{model_name}'
				if (exists(model_file)):
					t = {
						"UNL_HOME_2019": para_home2019,
						"UNL_EAST_2019": para_east2019,
						"UNL_KELLY_2019": para_kelly2019,
						"UNL_LINKS_2019": para_links2019,
						"UNL_JOHNSON_2020": para_johnson2020,
						"UNL_KELLY_2020": para_kelly2020,
						"UNL_HOME_2020": para_home2020,
						"UNL_NORTH_2020": para_north2020,
					}
					for a, b in t.items():
						test(b, model_file, UPDATE_HRLDAS, USE_FORECASTED_RAINFALL, alpha)

					test(value, model_file, UPDATE_HRLDAS, USE_FORECASTED_RAINFALL,alpha)
					break

def set_env():
	# print(device_lib.list_local_devices())
	# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	return

if __name__ == '__main__':

	main(sys.argv[1:])
	# set_env()
