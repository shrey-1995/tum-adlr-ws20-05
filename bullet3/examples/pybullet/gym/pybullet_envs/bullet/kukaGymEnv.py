import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
VISITING_CIRCLE_REWARD = 100
FINISHING_REWARD = 500
SPARSE = False
class KukaGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=1000):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self.seed()
    ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.Pos = (random.random(), random.random(), random.random())
    self.Pos1 = (random.random(), random.random(), random.random())
    self.Pos2 = (random.random(), random.random(), random.random())
    '''
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), (random.random(), random.random(), random.random()),
                               (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    self.blockUid1 = p.loadURDF(os.path.join(self._urdfRoot, "block1.urdf"), (random.random(), random.random(), random.random()),
                                (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    self.blockUid2 = p.loadURDF(os.path.join(self._urdfRoot, "block2.urdf"), (random.random(), random.random(), random.random()),
                                (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    '''
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      #TODO: correct the action high and low space
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):
    #print("KukaGymEnv _reset")
    self.visit_next = 0
    self.visited = np.zeros(3)
    self.terminated = 0
    self.done = False
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
              0.000000, 0.000000, 0.0, 1.0)

    xpos = 0.5
    ypos = 0.2
    ang = 3.14 * 0.5
    orn = p.getQuaternionFromEuler([0, 0, ang])


    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), (xpos+0.2, ypos, 0.15),
                               (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    self.blockUid1 = p.loadURDF(os.path.join(self._urdfRoot, "block1.urdf"), (xpos+0.3, 0.1+ypos, 0.15),
                                (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    self.blockUid2 = p.loadURDF(os.path.join(self._urdfRoot, "block2.urdf"), (xpos+0.3, ypos-0.1, 0.15),
                                (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    #self.agent = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), (-0.2 + random.random(), -0.2 + random.random(), -0.2 + random.random()),
    #                            (orn[0], orn[1], orn[2], orn[3]), useFixedBase=True)
    p.setGravity(0, 0, 0)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    #agentState = p.getBasePositionAndOrientation(self.agent)
    #agentPos = agentState[0]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    blockPos1, blockOrn1 = p.getBasePositionAndOrientation(self.blockUid1)
    blockPos2, blockOrn2 = p.getBasePositionAndOrientation(self.blockUid2)
    #TODO: check distance calculation
    #self.prev_dist = np.zeros(4)
    #self.prev_dist[0] = math.sqrt(math.pow(agentPos[0] - blockPos[0], 2) + math.pow(agentPos[1] - blockPos[1], 2) + math.pow(agentPos[2] - blockPos[2], 2))
    #self.prev_dist[1] = math.sqrt(math.pow(agentPos[0] - blockPos1[0], 2) + math.pow(agentPos[1] - blockPos1[1], 2) + math.pow(agentPos[2] - blockPos1[2], 2))
    #self.prev_dist[2] = math.sqrt(math.pow(agentPos[0] - blockPos2[0], 2) + math.pow(agentPos[1] - blockPos2[1], 2) + math.pow(agentPos[2] - blockPos2[2], 2))
    # Compute mindist

    self._envStepCounter = 0
    p.stepSimulation()

    #self._observation = list(agentPos) + self.getExtendedObservation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)*5

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    #change
    self._observation = self._kuka.getObservation()
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    blockPos1, blockOrn1 = p.getBasePositionAndOrientation(self.blockUid1)
    blockPos2, blockOrn2 = p.getBasePositionAndOrientation(self.blockUid2)

    #TODO : check, we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPos[0], blockPos[1], blockPos[2], blockPos1[0], blockPos1[1], blockPos1[2],
                                blockPos2[0], blockPos2[1], blockPos2[2]]

    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

    return list(self._observation) + list(blockInGripperPosXYEulZ)

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.005
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    else:
      #print("action[0]=", str(action[0]))
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      da = action[2] * 0.05
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    return self.step2(realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      #change
      self._kuka.applyAction(action)
      #agentPos_, agentOrn = p.getBasePositionAndOrientation(self.agent)
      #agentPos = (max(-0.9, min(1, agentPos_[0])), max(-0.9, min(1, agentPos_[1])), max(-0.9, min(1, agentPos_[2])))
      #action = action
      #agentPos = agentPos + action[:3]
      #p.resetBasePositionAndOrientation(self.agent, agentPos, agentOrn)
      p.stepSimulation()
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()
    if math.isnan(self._observation[0]):
      print("why")
    gripperPos = self._kuka.fingerPos

    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    blockPos1, blockOrn1 = p.getBasePositionAndOrientation(self.blockUid1)
    blockPos2, blockOrn2 = p.getBasePositionAndOrientation(self.blockUid2)
    curr_dist = np.zeros(4)
    #TODO: check distance calculation
    curr_dist[0] = math.sqrt(math.pow(gripperPos[0] - blockPos[0], 2) + math.pow(gripperPos[1] - blockPos[1], 2) + math.pow(gripperPos[2] - blockPos[2], 2))
    curr_dist[1] = math.sqrt(math.pow(gripperPos[0] - blockPos1[0], 2) + math.pow(gripperPos[1] - blockPos1[1], 2) + math.pow(gripperPos[2] - blockPos1[2], 2))
    curr_dist[2] = math.sqrt(math.pow(gripperPos[0] - blockPos2[0], 2) + math.pow(gripperPos[1] - blockPos2[1], 2) + math.pow(gripperPos[2] - blockPos2[2], 2))
    #diff = (self.prev_dist - curr_dist) * 1000
    diff = -curr_dist * 10
    # Init variables
    step_reward = np.zeros(4)
    current_visit = np.zeros(3)
    self.prev_dist = curr_dist
    ins_val = 0
    maxDist = 0.005

    #closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, maxDist)
    #closestPoints1 = p.getClosestPoints(self.blockUid1, self._kuka.kukaUid, maxDist)
    #closestPoints2 = p.getClosestPoints(self.blockUid2, self._kuka.kukaUid, maxDist)

    """
    closestPoints = p.getContactPoints(self.blockUid, self.agent)
    closestPoints1 = p.getContactPoints(self.blockUid, self.agent)
    closestPoints2 = p.getContactPoints(self.blockUid2, self.agent)
    """

    #TODO: check whether the correct way to detect contact between finger and block
    #contactPoints = p.getContactPoints(self.blockUid, self._kuka.kukaUid)
    #contactPoints1 = p.getContactPoints(self.blockUid1, self._kuka.kukaUid)
    #contactPoints2 = p.getContactPoints(self.blockUid2, self._kuka.kukaUid)

    contact = False if curr_dist[0] > 0.1 else True
    contact1 = False if curr_dist[1] > 0.1 else True
    contact2 = False if curr_dist[2] > 0.1 else True

    if (contact):
      ins_val = 1
      intersection = 0
    elif (contact1):
      ins_val = 1
      intersection = 1
    elif (contact2):
      ins_val = 1
      intersection = 2

    if not SPARSE:
      step_reward += diff

    if ins_val:
      current_visit[intersection] = 1
      step_reward[intersection] += VISITING_CIRCLE_REWARD
      if intersection == self.visit_next:
        # Preempt task on reaching the circle
        self.visited[intersection] = 1
        step_reward[3] = VISITING_CIRCLE_REWARD * (intersection + 1)
        self.visit_next += 1
        if np.sum(self.visited) == len(self.visited):
          self.done = True
          step_reward[3] = FINISHING_REWARD
          print("Done with reward: ", step_reward[3])

        print("Circle reached: ", step_reward)

      elif intersection == self.visit_next - 1:
        pass

      else:
        self.visit_next = 0
        step_reward[3] = -300

        for i in range(len(self.visited)):
          self.visited[i] = 0

        if intersection == 0:
          self.visited[0] = 1
          step_reward[3] += VISITING_CIRCLE_REWARD

    # Update observation space
    self._observation = self._observation + list(self.visited) + list(current_visit)
    #self._observation = list(agentPos) + self._observation  + list(self.visited) + list(current_visit)
    #print("self._envStepCounter")
    #print(self._envStepCounter)

    #done = self._termination()
    #npaction = np.array([
     #   action[3]
    #])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    #actionCost = np.linalg.norm(npaction) * 10.
    #print("actionCost")
    #print(actionCost)
    #reward = self._reward() - actionCost
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation)*5, step_reward, self.done, current_visit

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    #TODO: change logic
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                       self._kuka.kukaEndEffectorIndex)

    reward = -1000

    numPt = len(closestPoints)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      reward = reward + 10000
      print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
    #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

  def get_state_size(self):
    #return len(self.getExtendedObservation())+9
    return len(self.getExtendedObservation())+6
