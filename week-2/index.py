import math

print('==== Task1 ====')
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_from(self, other):
        return math.atan2(self.y - other.y, self.x - other.x)


class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def slope(self):
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    def is_parallel(self, other):
        return self.slope() == other.slope()

    def is_perpendicular(self, other):
        return self.slope() * other.slope() == -1

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def area(self):
        return math.pi * self.radius**2

    def intersects_with(self, other):
        return self.center.distance(other.center) < self.radius + other.radius

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def order_clockwise(self):
        reference_point = min(self.vertices, key=lambda p: (p.y, p.x))
        self.vertices.sort(key=lambda p: p.angle_from(reference_point))
        
        return self.vertices

    def perimeter(self):
        ordered_vertuces = self.order_clockwise()
        perimeter = 0
        for i in range(len(ordered_vertuces)):
            perimeter += ordered_vertuces[i].distance(ordered_vertuces[0 if i == len(ordered_vertuces) -1 else (i + 1)])
        return perimeter

p1_a, p2_a = Point(-6, 1), Point(2, 4)
p1_b, p2_b = Point(-6, -1), Point(2, 2)
p1_c, p2_c = Point(-4, -4), Point(-1, 6)

line_a = Line(p1_a, p2_a)
line_b = Line(p1_b, p2_b)
line_c = Line(p1_c, p2_c)

circle_a = Circle(Point(6, 3), 2)
circle_b = Circle(Point(8, 1), 1)

polygon_a = Polygon([Point(2, 0), Point(5, -1), Point(4, -4), Point(-1, -2)])

print('Are lineA and lineB parallel?', line_a.is_parallel(line_b))
print('Are lineC and line  perpendicular?', line_a.is_perpendicular(line_c))
print('Area of circleA:', circle_a.area())
print('Do circleA and circleB intersect?', circle_a.intersects_with(circle_b))
print('Perimeter of polygonA:', polygon_a.perimeter())

print('==== Task2 ====')
class Enemy:
    def __init__(self, label, x, y, vector_x, vector_y, life_points = 10):
        self.label = label
        self.x = x
        self.y = y
        self.vector_x = vector_x
        self.vector_y = vector_y
        self.life_points = life_points
        self.is_dead = False

    def move(self):
        if self.is_dead == False:
            self.x += self.vector_x
            self.y += self.vector_y

    
    def take_damage(self, damage):
        if self.is_dead  == False:
            self.life_points -= damage
            if self.life_points <= 0:
                self.is_dead = True
    
    def get_position(self):
        return (self.x, self.y)


class Tower:
    def __init__(self, x, y, attack_points = 1, attack_range = 2):
        self.x = x
        self.y = y
        self.attack_points = attack_points
        self.attack_range = attack_range

    def is_in_range(self, enemy):
        distance = math.sqrt((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2)
        return distance <= self.attack_range

    def attack(self, enemies):
        for enemy in enemies:
            if enemy.is_dead == False and self.is_in_range(enemy):
                enemy.take_damage(self.attack_points)


class AdvancedTower(Tower):
    def __init__(self, x, y):
        super().__init__(x, y, attack_points = 2, attack_range = 4)


class Game:
    def __init__(self):
        self.enemies = [
            Enemy('E1', -10, 2, 2, -1),
            Enemy('E2', -8, 0, 3, 1),
            Enemy('E3', -9, -1, 3, 0)
        ]
        
        self.basic_towers = [
            Tower(-3, 2),
            Tower(-1, -2),
            Tower(4, 2),
            Tower(7, 0)
        ]
        
        self.advanced_towers = [
            AdvancedTower(1, 1),
            AdvancedTower(4, -3)
        ]

    def run_turn(self):
        for enemy in self.enemies:
            enemy.move()
        
        for tower in self.basic_towers + self.advanced_towers:
            tower.attack(self.enemies)

    def print_enemy_status(self):
        for enemy in self.enemies:
            print(f'{enemy.label} - position: ({enemy.x}, {enemy.y}) - rest life points: {0 if enemy.is_dead else enemy.life_points}')



game = Game()
    
for turn in range(1, 11):
    game.run_turn()

game.print_enemy_status()