standard_prompt = '''Find the number of intersection points between the shapes and lines specified by the coordinates given.

input:There is a circle centered at (-1.5, -1.0) with radius 3.0. 
There is a polygon with coordinates [(-2.2, 4.0), (-3.2, -0.4), (2.4, -2.6), (3.8, 4.1)]. 
There is a line segment from (0.9, 3.3) to (-1.3, 3.4). 
How many intersection points are there?
output:2

input:{problem}
output:'''

cot_prompt = '''Find the number of intersection points between the shapes and lines specified by the coordinates given.

input:There is a circle centered at (-1.5, -1.0) with radius 3.0. 
There is a polygon with coordinates [(-2.2, 4.0), (-3.2, -0.4), (2.4, -2.6), (3.8, 4.1)]. 
There is a line segment from (0.9, 3.3) to (-1.3, 3.4). 
How many intersection points are there?
output:
1. circle(1) and polygon(2): line segment [(-2.2, 4.0), (-3.2, -0.4)] has 1 intersection with circle, line segment [(-3.2, -0.4), (2.4, -2.6)] has 1 intersection with circle, line segment [(2.4, -2.6), (3.8, 4.1)] has 0 intersection with circle, line segment [(3.8, 4.1), (-2.2, 4.0)] has 0 intersection with circle.
2. circle(1) and line segment(3): line segment[(0.9, 3.3), (-1.3, 3.4)] has 0 intersection with circle.
3. polygon(2) and line segment(3): line segment [(-2.2, 4.0), (-3.2, -0.4)] has 0 intersection with line segment, line segment [(-3.2, -0.4), (2.4, -2.6)] has 0 intersection with line segment, line segment [(2.4, -2.6), (3.8, 4.1)] has 0 intersection with line segment, line segment [(3.8, 4.1), (-2.2, 4.0)] has 0 intersection with line segment.
answer:2

input:{problem}
output:
'''
