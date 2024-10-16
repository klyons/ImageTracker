import numpy as np


class Line:
	def __init__(self, start_x, start_y, end_x, end_y) -> None:
		self.start = np.array([start_x, start_y])
		self.end = np.array([end_x, end_y])

class Point:
	def __init__(self) -> None:
		self.pt = np.array([x, y])

	def update(self, new_x, new_y):
		self.pt = np.array([new_x, new_y])

class point_slide(line, point):
    pass