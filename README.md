# Humanoid_RL
RL Implementation for a Poppy Torso to pick and place objects

Please note that the simulations need to be done on Mujoco, and for that you need a valid license. If you are a student, you can get a free license, using your Institution mail ID.

Contents:
1) Hardware: 
The Solidworks parts needed to assemble the Poppy model, and the final assembly needed to expirement the Reinforcement Learning algorithm are readily available. If you just want to proceed with the same Humanoid robot, you just need to load the model XML file into Mujoco
2) Environment: This has been implemented in both C++ and python. This contains the instructions regarding the states, actions, step, reward function, and reset states functionalities for the robot.
3) Reinforcement Learning model: We have tested the two approaches, Deep Deterministic Policy Gradients (DDPG) and Proximal Policy Optimization (PPO). We have experienced that DDPG, though slower, converges to a optimim policy; being an off policy algorithm. PPO faces convergence issues and generally got stuck in a local optimum; being an on policy approach.
