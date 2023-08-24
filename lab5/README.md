## env setup

```
pip install -r requirements.txt
```

## training

```
# LunarLander using DQN
python dqn.py --logdir log/dqn

# LunarLandar using DDPG
python ddpg.py --logdir log/ddpg

# Breakout using DQN
python dqn_breakout.py --logdir dqn_breakout
```

## testing
```
add '--test_only', and load ckpt with '-tmp' config
```