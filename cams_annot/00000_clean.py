import os

for f in os.listdir():
	if not os.path.isdir(f) and not '.py' in f:
		name, ext = os.path.splitext(f)
		if ext != '.xml':
			if not os.path.exists(name + '.xml'):
				os.remove(f)
			