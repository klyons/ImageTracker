import geometry


class IDGenerator:
	def get_id(self):
		self.next_id += 1
		return self.next_id


idgen = IDGenerator()
line1 = geometry.Line(372, 500, 1324, 500)
line1 = geometry.Line(1324, 500, 1801, 1080)

class Object:
	def __init__(self, x, y) -> None:
		self.point = geometry.Point(x, y)
		self.id = idgen.get_id()
		self.remaining_frames = 30
		self.fell = False

	def __eq__(self, other):
		if not isinstance(other, Object):
			return False
		return self.id == other.id

	def __str__(self):
		return f"Object {self.id} at ({self.remaining_frames} / 30)"

	def get_distance(self, target_x, target_y):
		return ((target_x - self.point.pt[0]) ** 2 + (target_y - self.point.pt[1]) ** 2) ** 0.5

	def can_remove(self):
		self.remaining_frames -= 1
		return self.remaining_frames == 0
 
	def update(self, new_x, new_y):
		newpt = geometry.Point(new_x, new_y)
		self.point.update(new_x, new_y)
		
		object_start_side_line = geometry.point_slide(line1, self.point)
		object_end_side_line = geometry.point_slide(line1, newpt)
  
		self.point = newpt
		self.remaining_frames = 30			
