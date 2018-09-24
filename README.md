# Humanoid_RL
RL Implementation for a Poppy Torso to pick and place objects

Steps Completed:
1. Created a Solidworks model for Poppy Torso with Grippers.
2. Created the xml file of the Robot for use in Mujoco
3. Fine tuned the parametes of the xml to minimize the error between control and qpos angles. The parameters kp = 250, damping = 1, and stiffness = 1 are close to the sweet spot; in this case.
4. Created the environment for the reach phase for Poppy.

Next Steps:
1. Check the environment model.
2. Train Poppy for the Reach phase using PPO or some other RL algorithm.

