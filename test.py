import gym
import gym_stocks
import random

env = gym.make('Stocks-v0')
# print(env.reset())
env.reset()

for i in range(10):
	prs = (random.randint(0,20)-10)/10
	data,reward,done, _ = env.step(prs)
	# print(data)
	print("act: {}, roi(reward): {}".format(prs,reward))
	print("---")
	#print env.step(0)
