from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class MTEnv(MiniGridEnv):
    """
    Environment in which coloured tiles with varing rewards 
    are placed in a gridworld. The a static goal is also present
    """

    def __init__(
        self,
        size=12,
        numObjs=9,
        n_colours=3,
        agent_start_pos=(1,1),
        agent_start_dir=1,
        goal_start_pos = None,
        static_tiles=True

    ):
        self.numObjs = numObjs
        # don't include blue (default colour) or green (goal colour)
        self.tile_colours = ['red', 'purple', 'yellow', 'grey']
        assert n_colours <= len(self.tile_colours) and n_colours > 0, "up to 4 colours can be used"    


        self.agent_start_pos= agent_start_pos
        self.agent_start_dir= agent_start_dir
        self.goal_start_pos = goal_start_pos

        self.tile_colours = self.tile_colours[:n_colours]
        self.tile_rewards = {col: random.choice([-1, 0, 1]) for col in self.tile_colours}
        self.positions_stale = True
        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=False
        )

    def _gen_grid(self, width, height):
        if self.positions_stale:
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height-1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width-1, 0)

            types = ['floor']
            objs = [] 

            # For each object to be generated
            while len(objs) < self.numObjs:
                objColor = random.choice(self.tile_colours)
                obj = Floor(objColor)

                self.place_obj(obj)
                objs.append(obj)
            
            self.positions_stale = False

        # Randomize the player start position and orientation
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        if self.goal_start_pos is None:
            self.put_obj(Goal(), width - 2, height - 2)
        else:
            self.place_obj(Goal())

        # Choose a random object to be picked up
        self.mission = "Reach the goal"
       
    def set_tile_rewards(self, tile_rewards: dict = None):
        self.reset()

        if tile_rewards == None:
            self.tile_rewards = {col: random.choice([-1, 0, 1]) for col in self.tile_colours}
        else:
            self.tile_rewards = tile_rewards

    def set_wall_colour(self, colour = None):
        for obj in self.grid.grid:
            if obj is not None:
                if obj.type == "wall":
                    obj.color = colour

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        
        current_obj = self.grid.get(*self.agent_pos)
        if current_obj is not None:  
            if current_obj.color in self.tile_colours and current_obj.type == "floor":
                reward += self.tile_rewards[current_obj.color]
                current_obj.color = "blue"
            
        return obs, reward, done, info

    def possible_object_rewards(self):
        # TODO can make this more efficent by keeping track of the tiles
        # will need to reinit whatever tracker is being
        # used when set_wall_colour is called

        max_obj_r = 0
        for obj in self.grid.grid:
            if obj is not None:
                if obj.type == "floor" and obj.color in self.tile_colours:
                    max_obj_r += np.max([0,self.tile_rewards[obj.color]])

        return max_obj_r


class MTEnvFourRoomsStatic(MTEnv):
    """
    Static FourRooms
    """
    def __init__(self):
        self._agent_default_pos = (1,1)  
        self._goal_default_pos = (17,17)

        super().__init__(size=19)

    def _gen_grid(self, width, height):
        # Create the grid
        if self.positions_stale:
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)

            room_w = width // 2
            room_h = height // 2

            # For each row of rooms
            for j in range(0, 2):

                # For each column
                for i in range(0, 2):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h

                    # Bottom wall and door
                    if i + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)

                    # Bottom wall and door
                    if j + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)
            types = ['floor']
            objs = [] 

            # For each object to be generated
            while len(objs) < self.numObjs:
                objColor = random.choice(self.tile_colours)
                obj = Floor(objColor)

                self.place_obj(obj)
                objs.append(obj)

            self.positions_stale = False

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = 0 
        else:
            self.place_agent()
            
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"

class MTEnvFourRoomsStaticWalls(MTEnv):
    """
    Static FourRooms
    """
    def __init__(self):
        self._agent_default_pos = (1,1)  
        self._goal_default_pos = (17,17)

        super().__init__(size=19)

    def _gen_grid(self, width, height):
        # Create the grid
        if self.positions_stale:
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)

            room_w = width // 2
            room_h = height // 2

            # For each row of rooms
            for j in range(0, 2):

                # For each column
                for i in range(0, 2):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h

                    # Bottom wall and door
                    if i + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)

                    # Bottom wall and door
                    if j + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)
            types = ['floor']
            objs = [] 

            # For each object to be generated
            while len(objs) < self.numObjs:
                objColor = random.choice(self.tile_colours)
                obj = Floor(objColor)

                self.place_obj(obj)
                objs.append(obj)

            self.positions_stale = False

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = 0 
        else:
            self.place_agent()
            
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"



class MTEnvFourRoomsShufflePositions(MTEnv):
    def __init__(self):
        self._agent_default_pos = (1,1)  
        self._goal_default_pos = (17,17)
        super().__init__(size=19)

    def _gen_grid(self, width, height):
        # Create the grid
        if self.positions_stale:
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)

            room_w = width // 2
            room_h = height // 2

            # For each row of rooms
            for j in range(0, 2):

                # For each column
                for i in range(0, 2):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h

                    # Bottom wall and door
                    if i + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)

                    # Bottom wall and door
                    if j + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)
            types = ['floor']
            objs = [] 

            # For each object to be generated
            while len(objs) < self.numObjs:
                objColor = random.choice(self.tile_colours)
                obj = Floor(objColor)

                self.place_obj(obj)
                objs.append(obj)

            self.positions_stale = False

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = 0 
        else:
            self.place_agent()
            
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"

    def set_tile_rewards(self, tile_rewards: dict = None):
        # set postitions stale to cause a regeneration
        self.positions_stale = True
        self.reset()

        if tile_rewards == None:
            self.tile_rewards = {col: random.choice([-1, 0, 1]) for col in self.tile_colours}
        else:
            self.tile_rewards = tile_rewards


class MTEnvFourRoomsLandmarks(MTEnv):
    def __init__(self):
        self._agent_default_pos = (1,1)  
        self._goal_default_pos = (17,17)
        self.n_landmarks = 2
        super().__init__(size=19)

    def _gen_grid(self, width, height):
        # Create the grid
        if self.positions_stale:
            self.grid = Grid(width, height)

            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)

            room_w = width // 2
            room_h = height // 2

            # For each row of rooms
            for j in range(0, 2):

                # For each column
                for i in range(0, 2):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h

                    # Bottom wall and door
                    if i + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)

                    # Bottom wall and door
                    if j + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)

            types = ['floor']
            objs = [] 
            # place landmarks
            valid_idx_x = [i for i in range(2,8)] + [i for i in range(11,17)]
            valid_idx_y = [i for i in range(2,8)] + [i for i in range(11,17)]
            for n in range(self.n_landmarks):
                self.put_obj(Wall(),random.choice(valid_idx_x), random.choice(valid_idx_y))
            # For each object to be generated
            while len(objs) < self.numObjs:
                objColor = random.choice(self.tile_colours)
                obj = Floor(objColor)

                self.place_obj(obj)
                objs.append(obj)

            self.positions_stale = False

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = 0 
        else:
            self.place_agent()
            
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"



        
class MTEnv8x8N9(MTEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=9)

register(
        id='MiniGrid-MTEnv-8x8-v0',
        entry_point='gym_minigrid.envs:MTEnv8x8N9'
)

#register(
#        id='MiniGrid-MTEnvFourRooms-v0',
#        entry_point='gym_minigrid.envs:MTEnvFourRooms'
#)

register(
        id='MiniGrid-MTEnvFourRoomsStatic-v0',
        entry_point='gym_minigrid.envs:MTEnvFourRoomsStatic'
)
register(
    id='MiniGrid-MTEnvFourRoomsStaticWalls-v0',
        entry_point='gym_minigrid.envs:MTEnvFourRoomsStaticWalls'
)
register(
        id='MiniGrid-MTEnvFourRoomsShufflePositions-v0',
        entry_point='gym_minigrid.envs:MTEnvFourRoomsShufflePositions'
)
register(
        id='MiniGrid-MTEnvFourRoomsLandmarks-v0',
        entry_point='gym_minigrid.envs:MTEnvFourRoomsLandmarks'
)
