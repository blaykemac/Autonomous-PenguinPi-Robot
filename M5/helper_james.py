import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

###### OLD MATH FUNCTIONS ######
def compute_distance_between_points(p1, p2):
    """ 
        Computes distance between two points
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.hypot(dx, dy)


def is_point_in_segment(start_seg, end_seg, point_q):
    """ 
        Determines in point_q is strictly inside the segment defined by start_seg and end_seg
    """
    dist_1 = compute_distance_between_points(start_seg, point_q)
    dist_2 = compute_distance_between_points(end_seg, point_q)
    dist_3 = compute_distance_between_points(start_seg, end_seg)
     
    return dist_1 + dist_2 == dist_3


def compute_lines_intersection(line_1, line_2):
    
    """ 
    The orthogonal projection of a point onto a line is defined as the
    intersection between two perpendicular lines (with the point of
    interest being along one of the lines)
    
    This orthogonal projection can be obtained by finding the inteserction
    point between 2 perpendicular lines
       
    Lines are defined in standard form ax + by + c = 0
    line_1 = <a, b, c>
    line_2 = <a1, b1, c1>
    """
    d  = line_1[0] * line_2[1] - line_1[1] * line_2[0]
    dx = line_1[2] * line_2[1] - line_1[1] * line_2[2]
    dy = line_1[0] * line_2[2] - line_1[2] * line_2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return np.array([x,y])
    return False


def compute_line_through_points(p1, p2):
   
    """ 
    Computes line defined by 2 points
    Line is returned in standard form ax + by + c = 0
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # If p1 and p2 are different, compute the line defined by these 2 points
    if np.all(np.isclose(p1, p2)):
        return False
    
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0]*p2[1] - p2[0]*p1[1]

    return [a, b, -c]


def compute_distance_point_to_line_by_intersection(start_line, end_line, point_q):
    
    """
    Computes distance from point_q and the line defined by 
    (start_line, end_line).
    
    This corresponds to finding the closest point (x0,y0) on the line to point_q
    (i.e, orthogonal projection of point_q onto the line) and computing
    the distance between (x0,y0) and point_q
    
    (x0, y0) coordinates are computed by considering the intersection between
    two perpendicular lines. The first line is defined by the points start_line
    and end_line. The second line is perpendicular to the first one and goes through
    point_q
    
    """

    # Put point_q in right format
    point_q = np.array(point_q)
    
    # Compute first line parameters
    first_line = compute_line_through_points(start_line, end_line)
           
    if first_line:
        a, b, c = first_line
        
        # Normalize parameters so that a*a + b*b = 1
        ab_norm = np.linalg.norm(np.array([a, b]))
        a_norm = a / ab_norm
        b_norm = b / ab_norm
        
        # Define a norm vector perpendicular to first line
        # This vector is parallel to the line that goes through point_q
        normal = np.array([a_norm, b_norm])

        # Using the vector-form equation of a line, to find a second in line 2
        point_q2 = point_q + 2*(normal)
        
        # Using point_q and point_q2, find the standard parameters of second line
        second_line = compute_line_through_points(point_q, point_q2)
        if second_line:

            # Compute intersection point
            proj_point = compute_lines_intersection(first_line, second_line)
            
            if proj_point is not None:
                # Compute distance between point_q and its projection onto first line
                distance = compute_distance_between_points(proj_point, point_q)
                return proj_point, distance
            
    return False


def compute_distance_point_to_segment(start_seg, end_seg, point_q):
    """
    Computes distance from point_q and line segment defined by start_seg and end_seg
    
    This method first computes the distance (and othorgonal projection of point_q) 
    to the line defined by start_seg and end_seg.
    
    If proj_point_q is stricly inside the segment, it returns the distance
    to the line and the indicator w=1
    
    If proj_point_q is not in the segment, it determines the closest segment
    point to point_q, computes the distance between point_q and the chosen segment point
    
    The indicator will be set to w=1 if start_seg is the closest. Otherwise,
    end_seg is the closest point and w=2
    
    """
    
    # Compute point_q projection and distance from point_q to projection
    proj_q, distance = compute_distance_point_to_line_by_intersection(start_seg, end_seg, point_q)
    
    # Compute distance to start and end of segment
    dist_to_start = compute_distance_between_points(start_seg, point_q)
    dist_to_end = compute_distance_between_points(end_seg, point_q)
    
    if is_point_in_segment(start_seg, end_seg, proj_q):
        w = 0  # orthogonal projection is the closest point
    elif dist_to_start < dist_to_end:
        w = 1 # start is the closest point in segment
        distance = dist_to_start
        proj_q = start_seg
    else:
        w = 2 # end is the closest point in segment
        distance = dist_to_end
        proj_q = end_seg
                
    return w, distance, proj_q


def get_direction_from_points(p1, p2):
    """
    Computes horizontal angle between line defined by p1 and p2 and world x-axis
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.atan2(dy, dx)
    return np.array([np.cos(angle), np.sin(angle)])


def get_direction_from_line(line):
    """
    Computes direction of vector parallel to standard line ax + by + c = 0
    """
    a, b, c = line
    vector = np.array([b, -a])
    vector = vector / np.linalg.norm(vector)
    return vector


def point_in_line(line, point):
    """
    Determines if a point is strictly inside a line
    """
    a, b, c = line
    return np.isclose(a*point[0] + b*point[1] - c, 0)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.linalg.norm(array - value, axis=1)).argmin()
    return idx, array[idx]
    
######END######

###### OLD OBSTACLE FILE #######

class Polygon:
	"""
	Obstacles are represented as polygons
	Polygons are defined as an array with n rows (vertices) and 2 columns
	
	"""

	def __init__(self, vertices=np.zeros((4,2))):
		self.vertices = vertices
	
	def compute_distance_point_to_polygon(self, point_q):
		"""
		Compute distance from point_q to the closest point in the polygon
	
		Method returns:
		- dist: minimal distance from point_q to polygon
		- indices of segment closest to point_q

		"""        
		dist = np.inf
		segment_idx = None
		closest_point = None
		len_polygon = self.vertices.shape[0]
		
		for i in range(len_polygon):
			case, seg_dist, _ = compute_distance_point_to_segment(self.vertices[i],
																  self.vertices[(i+1) % len_polygon], point_q)
			if seg_dist <= dist:
				dist = seg_dist
				if case == 0:
					closest_point = i
				elif case == 1:
					closest_point = i
				else:
					closest_point = (i+1) % len_polygon                
		
		segment_idx = (closest_point, (closest_point+1) % len_polygon)
		# print("Closest segment is {}, {}".format(segment_idx[0], segment_idx[1]))
		return dist, segment_idx
			
	def compute_tangent_vector_to_polygon(self, point_q, idx):  
		
		"""
		Determines the unit-length vector tangent at point_q to the polygon
			
		Method returns:
		   tangent vector

		"""  
		   
		v1 = self.vertices[idx[0]]
		v2 = self.vertices[idx[1]]
			
		case, seg_dist, closest_point = compute_distance_point_to_segment(v1, v2, point_q)    
	
		tangent_vector = (v2-v1)/np.linalg.norm(v2-v1)
		
		return tangent_vector
	

	def plot_obstacle(self):
		ox = []
		oy = []

		for i in range(self.vertices.shape[0]):
			x0, y0 = self.vertices[i]
			x1, y1 = self.vertices[(i+1) % self.vertices.shape[0]]

			if x0 == x1:
				start = np.min([y0, y1])
				end = np.max([y0, y1])
				y_values = np.linspace(start, end, int(np.ceil((end-start)/0.2)), endpoint=True) 
				x_values = [x0] * len(y_values)
				ox.extend(x_values)
				oy.extend(y_values)

			if y0 == y1:
				start = np.min([x0, x1])
				end = np.max([x0, x1])
				x_values = np.linspace(start, end, int(np.ceil((end-start)/0.2)), endpoint=True) 
				y_values = [y0] * len(x_values)
				ox.extend(x_values)
				oy.extend(y_values)

		return ox, oy


	def to_display_format(self, screen_height):
		coordinates = [coordinates_to_pygame(v, screen_height) for v in self.vertices[0:-1]]
		return coordinates


	def is_in_collision_with_points(self, points, min_dist=2.5):

		# First check if point is within polygon
		points_in_collision = []

		for point in points:
			count_collisions = 0
			p_x, p_y = point
			for i in range(self.vertices.shape[0]-2):
				j = i - 1 if i != 0 else self.vertices.shape[0]-2
				v1 = self.vertices[i]
				v2 = self.vertices[j]

				if ((v1[1] > p_y) != (v2[1] > p_y)) and (p_x < (v2[0] - v1[0]) * (p_y - v1[1]) / (v2[1] - v1[1] + v1[0])):
					count_collisions += 1

			if count_collisions % 2 == 1:
				points_in_collision.append(point)
		
		if len(points_in_collision):
			return True

		# Second check if point is in collision with edges
		dist, _ = self.compute_distance_point_to_polygon(points[-1])
		if dist < min_dist:
			return True

		return False


	def get_perimeter(self):

		perimeter = 0

		for i in range(self.vertices.shape[0]-1):
			v1 = self.vertices[i]
			v2 = self.vertices[i+1]

			perimeter += compute_distance_between_points(v1, v2)

		return perimeter


class Rectangle(Polygon):
	
	def __init__(self, origin=np.zeros(2), width=100, height=20):
		self.width = width
		self.height = height
		self.origin = origin
				
		v1 = origin
		v2 = origin + np.array([width, 0])
		v3 = origin + np.array([width, -height])
		v4 = origin + np.array([0, -height])
		
		Polygon.__init__(self, vertices=np.array([v1, v2, v3, v4]))
		
	def to_display_format(self, screen_height):
		py_origin = coordinates_to_pygame(self.origin, screen_height)
		return (py_origin[0], py_origin[1], self.width, self.height)

	def plot_obstacle(self):
		return super().plot_obstacle()


class Circle:
	
	def __init__(self, c_x, c_y, radius):
		self.center = np.array([c_x, c_y])
		self.radius = radius

	def is_in_collision_with_points(self, points):

		dist = []
		for point in points:
			dx = self.center[0] - point[0]
			dy = self.center[1] - point[1]

			dist.append(dx * dx + dy * dy)

		if np.min(dist) <= self.radius ** 2:
			return True

		return False  # safe
        
######END######

###### OLD PATH ANIMATION ######


def plot_obstacles(obstacles, ax):
    for obs in obstacles:
        ox, oy = obs.plot_obstacle()
        ax.scatter(ox, oy, s=7, c='k')

def animate_path_bug1(initial_robot_pos,goal_pos,path,obstacles):
    # Set plot variables
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0, 50), ylim=(0, 60))
    ax.grid()

    goal, = ax.plot([], [], 'o', lw=2, c='g')
    path_line, = ax.plot([], [], 'r', lw=2)
    robot, = ax.plot([], [], 'ob', lw=3)
    plot_obstacles(obstacles, ax)
    steps_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Define the start and goal positions 
    g_x, g_y = goal_pos
    s_x, s_y = initial_robot_pos
    
    if len(path) == 0:
        print("Path was not found!!")
        ani = 0
        return
        
    else:
        # Animation code
        def init():
            goal.set_data(g_x, g_y)
            robot.set_data(s_x, s_y)
            path_line.set_data(s_x, s_y)
            steps_text.set_text('')
            return robot, goal, path_line, steps_text

        def animate(i):
            """perform animation step"""
            if i < path.shape[0]:
                pos = path[i,:]
                robot.set_data(pos[0], pos[1])
                path_line.set_data(path[:i, 0], path[:i, 1])
                steps_text.set_text('Steps: %.1f' % i)
            return robot, path_line

        ani = animation.FuncAnimation(fig, animate, frames=600, repeat=False, interval=10, init_func=init)
        return ani


def animate_path_bug2(initial_robot_pos,goal_pos,path,obstacles):
    # Set plot variables
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0, 50), ylim=(0, 60))
    ax.grid()

    goal, = ax.plot([], [], 'o', lw=2, c='g')
    line, = ax.plot([], [], 'r--', lw=1)
    path_line, = ax.plot([], [], 'g', lw=2)
    robot, = ax.plot([], [], 'ob', lw=4)
    steps_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    custom_lines = [Line2D([0], [0], color='r', lw=4)]
    ax.legend(custom_lines, ['start-goal line'])

    plot_obstacles(obstacles, ax)

    s_x, s_y = initial_robot_pos
    g_x, g_y = goal_pos
    
    if len(path)==0:
        print("Path was not found!!")
    else:
        # Animation code
        def init():
            goal.set_data(g_x, g_y)
            robot.set_data(s_x, s_y)
            path_line.set_data(s_x, s_y)
            steps_text.set_text('')
            line.set_data([s_x, g_x], [s_y, g_y])
            return robot, goal, path_line, steps_text

        def animate(i):
            """perform animation step"""
            pos = path[i,:]
            robot.set_data(pos[0], pos[1])
            path_line.set_data(path[:i, 0], path[:i, 1])
            steps_text.set_text('Steps: %.1f' % i)
            return robot, path_line

        ani = animation.FuncAnimation(fig, animate, frames=700, blit=True, interval=10, init_func=init)
        return ani
    
def plot_circle(ax, x, y, size, color="-b"):  
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        ax.plot(xl, yl, color)    
    
def animate_path_rrt(rrt, xlim1, ylim1):
    path = rrt.planning()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=xlim1, ylim=ylim1)
    ax.grid()

    ax.plot(rrt.start.x, rrt.start.y, "^r", lw=5)
    ax.plot(rrt.end.x, rrt.end.y, "^c", lw=5)

    obs_colour = ["r", "y", "k", "b"]
    
    for obs in rrt.obstacle_list:
        cx, cy = obs.center
        plot_circle(ax, cx, cy, obs.radius, color = obs_colour[obs.tag])

    for node in rrt.node_list:
        if node.parent:
            ax.plot(node.path_x, node.path_y, "r-")

    line_path, = ax.plot([], [], "b", lw=3)

    if path is not None:
        path_in_order = np.flipud(path)
        start_pos = path_in_order[0,:]

        def init():
            line_path.set_data(path_in_order[0], path_in_order[1])
            return line_path

        def animate(i):
            """perform animation step"""
            line_path.set_data(path_in_order[:i,0], path_in_order[:i, 1])
            return line_path

        ani = animation.FuncAnimation(fig, animate, frames=30, blit=True, interval=100, init_func=init)
        return ani
    else:
        print("Path was not found!!")
    
def animate_path_rrtc(rrtc):
    path = rrtc.planning()
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0, 16), ylim=(0, 12))
    ax.grid()

    ax.plot(rrtc.start.x, rrtc.start.y, "^r", lw=5)
    ax.plot(rrtc.end.x, rrtc.end.y, "^c", lw=5)
    
    
    
    for obs in rrtc.obstacle_list:
        cx, cy = obs.center
        plot_circle(ax, cx, cy, obs.radius)

    for node in rrtc.start_node_list:
        if node.parent:
            ax.plot(node.path_x, node.path_y, "g.-")

    for node in rrtc.end_node_list:
        if node.parent:
            ax.plot(node.path_x, node.path_y, "b.-")
            
    line_path, = ax.plot([], [], "r", lw=3)
    custom_lines = [Line2D([0], [0], color='g', lw=4), Line2D([0], [0], color='b', lw=4)]
    ax.legend(custom_lines, ['Start tree', 'End tree'])

    if path is not None:
        path = np.array(path)
        start_pos = path[0,:]
        
        def init():
            line_path.set_data(start_pos[0], start_pos[1])
            return line_path

        def animate(i):
            """perform animation step"""
            line_path.set_data(path[:i,0], path[:i, 1])
            return line_path

        ani = animation.FuncAnimation(fig, animate, frames=30, repeat=False, interval=100, init_func=init)
        return ani
    else:
        print("Path was not found!!")
        
def animate_path_prm(rmap,start, goal, num_samples, robot_size, max_distance, max_neighbours):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-10, 70), ylim=(-10, 70))
    plt.grid(True)

    obs = rmap.obstacles.data
    ax.plot(obs[:,0], obs[:,1], ".k")

    rmap.__generate_roadmap__(num_samples, max_distance, max_neighbours, robot_size)

    for i, v_edges in enumerate(rmap.edges):
        for e_idx in v_edges:
            v_from = rmap.vertices[i,:]
            v_to = rmap.vertices[e_idx,:]

            ax.plot([v_from[0], v_to[0]],
                    [v_from[1], v_to[1]], "k--", lw=0.5)
    
    path = rmap.plan(start, goal)

    # Animation code
    start_plot, = ax.plot([], [], "^r", lw=5)
    goal_plot, = ax.plot([], [], "^c", lw=5)
    line_path, = ax.plot([], [], "b", lw=2)    

    if path is not None:
        path_in_order = np.flipud(path)
        start_pos = path_in_order[0,:]

        def init():
            print(init)
            start_plot.set_data(start[0], start[1])
            goal_plot.set_data(goal[0], goal[1])
            line_path.set_data([], [])
            return line_path, start_plot, goal_plot

        def animate(i):
            """perform animation step"""
            line_path.set_data(path_in_order[:i,0], path_in_order[:i, 1])
            return line_path

        ani = animation.FuncAnimation(fig, animate, frames=30, blit=True, interval=30, init_func=init)
        return ani
    else:
        print("Path was not found!!")


###### RRT CLASS CODE ######

# This is an adapted version of the RRT implementation done by Atsushi Sakai (@Atsushi_twi)

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start=np.zeros(2),
                 goal=np.array([120,90]),
                 obstacle_list=None,
                 width = 160,
                 height=100,
                 expand_dis=3.0, 
                 path_resolution=0.5, 
                 max_points=500):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolion: step size to considered when looking for node to expand
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        while len(self.node_list) <= self.max_nodes:
            
            # 1. Generate a random node           
            rnd_node = self.get_random_node()
            
            # 2. Find node in tree that is closest to sampled node.
            # This is the node to be expanded (q_expansion)
            expansion_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            expansion_node = self.node_list[expansion_ind]

            # 3. Select a node (nearby_node) close to expansion_node by moving from expantion_node to rnd_node
            
            nearby_node = self.steer(expansion_node, rnd_node, self.expand_dis)

            # 4. Check if nearby_node is in free space (i.e., it is collision free). If collision free, add node
            # to self.node_list
            
            if self.is_collision_free(nearby_node):
                self.node_list.append(nearby_node)
                
            # If we are close to goal, stop expansion and generate path
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.is_collision_free(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Given two nodes from_node, to_node, this method returns a node new_node such that new_node 
        is “closer” to to_node than from_node is.
        """
        
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # How many intermediate positions are considered between from_node and to_node
        n_expand = math.floor(extend_length / self.path_resolution)

        # Compute all intermediate positions
        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_theta
            new_node.y += self.path_resolution * sin_theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node


    def is_collision_free(self, new_node):
        """
        Determine if nearby_node (new_node) is in the collision-free space.
        """
        if new_node is None:
            return True

        points = np.vstack((new_node.path_x, new_node.path_y)).T
        for obs in self.obstacle_list:
            in_collision = obs.is_in_collision_with_points(points)
            if in_collision:
                return False
        
        return True  # safe
        
    
    def generate_final_course(self, goal_ind):
        """
        Reconstruct path from start to end node
        """
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        x = self.width * np.random.random_sample() * (-1 if np.random.randint(0,2) else 1)
        y = self.height * np.random.random_sample() * (-1 if np.random.randint(0,2) else 1)
        rnd = self.Node(x, y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # Compute Euclidean disteance between rnd_node and all nodes in tree
        # Return index of closest element
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy) #returns the Euclidean norm
        theta = math.atan2(dy, dx)
        return d, theta        
    
    

class CircleT(Circle):
    def __init__(self, c_x, c_y, radius, tag):
        Circle.__init__(self, c_x, c_y, radius)
        self.tag = tag
