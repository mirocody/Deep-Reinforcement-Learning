import gym
env = gym.make('CartPole-v0')
env.reset()

# for i in range(100):
#     env.step(env.action_space.sample())
#     env.render()
#     screen = env.render(mode='rgb_array')
#     env.close()
#     # print (screen)
#     # print ("*****************************")
#     # screen = env.render(mode='rgb_array').transpose(
#     #     (2, 0, 1))  # transpose into torch order (CHW)
#     # print (screen)
#     # env.close()
screen = env.render(mode='rgb_array')
print (screen.shape)
screen = screen.transpose(
    (2, 0, 1))  # transpose into torch order (CHW)
print (len(screen))
print (screen.shape)
screen = screen[:, 160:320]
# print (screen)
# print (type(screen))
# print (len(screen[0]))


env.close()
