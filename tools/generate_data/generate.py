import pygame
import math
import random
import os
import csv

NUM_ITERATIONS = 10
NUM_FRAMES = 100
OUTPUT_DIR = 'longoutput'
SHOW = True

class Square:
    def __init__(self, x, y, angle=0):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = angle         # in radians
        self.omega = 0             # angular velocity (radians per second)
        self.size = 64             # side length of the square
        self.half_size = self.size / 2
        self.mass = 1.0
        # Moment of inertia for a square about its center: I = m*s^2/6.
        self.inertia = self.mass * (self.size ** 2) / 6.0

    def get_corners(self):
        """Return the world coordinates of the square's four corners in a sorted order."""
        corners = []
        # Compute the four corners relative to the center.
        for dx in (-self.half_size, self.half_size):
            for dy in (-self.half_size, self.half_size):
                rotated = pygame.math.Vector2(
                    dx * math.cos(self.angle) - dy * math.sin(self.angle),
                    dx * math.sin(self.angle) + dy * math.cos(self.angle)
                )
                corners.append(self.pos + rotated)
        # Sort corners in a clockwise order around the center.
        corners.sort(key=lambda p: math.atan2(p.y - self.pos.y, p.x - self.pos.x))
        return corners

    def update(self, dt, gravity):
        """Update position and rotation using simple Euler integration."""
        # Linear update under gravity.
        self.vel.y += gravity * dt
        self.pos += self.vel * dt
        # Update rotation.
        self.angle += self.omega * dt
        # Apply mild angular damping to reduce excessive spin.
        self.omega *= 0.99

    def resolve_collision(self, normal, contact_point, penetration):
        """Resolve a collision with a wall given its normal, the contact point, and penetration depth."""
        restitution = 0.9  # Coefficient of restitution.
        r = contact_point - self.pos  # Vector from center to contact point.
        # Velocity at the contact point (linear + rotational contributions).
        v_contact = self.vel + pygame.math.Vector2(-self.omega * r.y, self.omega * r.x)
        v_rel = v_contact.dot(normal)
        # Only resolve if the contact point is moving into the wall.
        if v_rel >= 0:
            return
        # Calculate impulse scalar: j = -(1+e) * v_rel / (1/m + (r x n)^2 / I).
        r_cross_n = r.x * normal.y - r.y * normal.x
        j = -(1 + restitution) * v_rel / (1 / self.mass + (r_cross_n ** 2) / self.inertia)
        impulse = j * normal
        # Update linear velocity.
        self.vel += impulse / self.mass
        # Scale down the angular impulse to reduce rotation tendency.
        self.omega += 0.5 * (r.x * impulse.y - r.y * impulse.x) / self.inertia
        # Positional correction to resolve penetration.
        self.pos += normal * penetration

# --- New Triangle class ---
class Triangle:
    def __init__(self, x, y, angle=0):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = angle
        self.omega = 0
        self.size = 64  # side length for an equilateral triangle
        self.mass = 1.0
        # Moment of inertia for an equilateral triangle about its centroid.
        self.inertia = self.mass * (self.size ** 2) / 12.0

    def get_corners(self):
        """Return the world coordinates of the triangle's three vertices."""
        corners = []
        # For an equilateral triangle, the distance from the centroid to a vertex.
        r = self.size / math.sqrt(3)
        # Vertices at 0, 120, 240 degrees relative to self.angle.
        for offset in (0, 2 * math.pi / 3, 4 * math.pi / 3):
            theta = self.angle + offset
            vertex = pygame.math.Vector2(r * math.cos(theta), r * math.sin(theta))
            corners.append(self.pos + vertex)
        return corners

    def update(self, dt, gravity):
        self.vel.y += gravity * dt
        self.pos += self.vel * dt
        self.angle += self.omega * dt
        self.omega *= 0.99

    def resolve_collision(self, normal, contact_point, penetration):
        restitution = 0.9
        r = contact_point - self.pos
        v_contact = self.vel + pygame.math.Vector2(-self.omega * r.y, self.omega * r.x)
        v_rel = v_contact.dot(normal)
        if v_rel >= 0:
            return
        r_cross_n = r.x * normal.y - r.y * normal.x
        j = -(1 + restitution) * v_rel / (1 / self.mass + (r_cross_n ** 2) / self.inertia)
        impulse = j * normal
        self.vel += impulse / self.mass
        self.omega += 0.5 * (r.x * impulse.y - r.y * impulse.x) / self.inertia
        self.pos += normal * penetration

# --- New Circle class ---
class Circle:
    def __init__(self, x, y, angle=0):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = angle  # Not used for collision but kept for consistency.
        self.omega = 0
        self.radius = 32  # Half of 64 for consistency.
        self.mass = 1.0
        # Moment of inertia for a disc: I = 0.5 * m * r^2
        self.inertia = 0.5 * self.mass * (self.radius ** 2)

    def get_corners(self):
        # Approximate circle with a 20-vertex polygon.
        points = []
        for i in range(20):
            theta = 2 * math.pi * i / 20
            vertex = pygame.math.Vector2(self.radius * math.cos(theta),
                                         self.radius * math.sin(theta))
            points.append(self.pos + vertex)
        return points

    def update(self, dt, gravity):
        self.vel.y += gravity * dt
        self.pos += self.vel * dt
        self.angle += self.omega * dt
        self.omega *= 0.99

    def resolve_collision(self, normal, contact_point, penetration):
        restitution = 0.9
        r = contact_point - self.pos
        v_contact = self.vel + pygame.math.Vector2(-self.omega * r.y, self.omega * r.x)
        v_rel = v_contact.dot(normal)
        if v_rel >= 0:
            return
        r_cross_n = r.x * normal.y - r.y * normal.x
        j = -(1 + restitution) * v_rel / (1 / self.mass + (r_cross_n ** 2) / self.inertia)
        impulse = j * normal
        self.vel += impulse / self.mass
        self.omega += 0.5 * (r.x * impulse.y - r.y * impulse.x) / self.inertia
        self.pos += normal * penetration

pygame.init()
window_size = 256
# use hidden mode to avoid rendering display
if SHOW:
    screen = pygame.display.set_mode((window_size, window_size))
else:
    screen = pygame.display.set_mode((window_size, window_size), pygame.HIDDEN)
clock = pygame.time.Clock()

# --- Updated spawn_shape ---
def spawn_shape():
    pos_x = random.uniform(32, window_size - 32)
    pos_y = random.uniform(32, window_size - 32)
    angle = random.uniform(0, 2 * math.pi)
    print(f"Spawning at ({pos_x}, {pos_y}) with angle {angle}")
    shape_type = random.choice([Square, Triangle, Circle])
    shape = shape_type(pos_x, pos_y, angle)
    shape.vel = pygame.math.Vector2(random.uniform(-1000, 1000), random.uniform(-1000, 100))
    shape.omega = random.uniform(-0.5, 0.5)
    return shape

# Gravity (pixels per second squared).
gravity = 9.81 * 100  

# Outer loop: 10000 iterations (each iteration simulates 20 frames)
for iteration in range(1, NUM_ITERATIONS + 1):
    # Create output folder for current iteration.
    out_folder = os.path.join(OUTPUT_DIR, str(iteration))
    os.makedirs(out_folder, exist_ok=True)
    
    # Open CSV file for the iteration.
    csv_path = os.path.join(out_folder, "data.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header.
        csvwriter.writerow(["frame", "shape", "x", "y", "angle", "vx", "vy", "angular_velocity"])
        
        # Spawn a new shape.
        shape = spawn_shape()
        
        # Simulate for 20 frames.
        for frame in range(1, NUM_FRAMES + 1):
            # Replace clock.tick with a constant dt for speed.
            if SHOW:
                dt = clock.tick(45) / 1000.0
            else:
                dt = 1.0 / 45.0
            
            # Optionally remove event processing if not needed:
            # for event in pygame.event.get():
            #    if event.type == pygame.QUIT:
            #         pygame.quit()
            #         exit()
            
            # Update simulation.
            shape.update(dt, gravity)
            corners = shape.get_corners()
            
            # --- Collision Detection and Response ---
            # Left wall.
            min_x = min(corner.x for corner in corners)
            if min_x < 0:
                contact_corner = min(corners, key=lambda c: c.x)
                penetration = 0 - contact_corner.x
                normal = pygame.math.Vector2(1, 0)
                shape.resolve_collision(normal, contact_corner, penetration)
            # Right wall.
            max_x = max(corner.x for corner in corners)
            if max_x > window_size:
                contact_corner = max(corners, key=lambda c: c.x)
                penetration = contact_corner.x - window_size
                normal = pygame.math.Vector2(-1, 0)
                shape.resolve_collision(normal, contact_corner, penetration)
            # Top wall.
            min_y = min(corner.y for corner in corners)
            if min_y < 0:
                contact_corner = min(corners, key=lambda c: c.y)
                penetration = 0 - contact_corner.y
                normal = pygame.math.Vector2(0, 1)
                shape.resolve_collision(normal, contact_corner, penetration)
            # Bottom wall.
            max_y = max(corner.y for corner in corners)
            if max_y > window_size:
                contact_corner = max(corners, key=lambda c: c.y)
                penetration = contact_corner.y - window_size
                normal = pygame.math.Vector2(0, -1)
                shape.resolve_collision(normal, contact_corner, penetration)
            
            # Render: clear screen, draw shape.
            screen.fill((255, 255, 255))
            pygame.draw.polygon(screen, (0, 0, 0), [(corner.x, corner.y) for corner in corners])
            pygame.display.flip()
            
            # Save frame image.
            image_path = os.path.join(out_folder, f"frame_{frame}.png")
            pygame.image.save(screen, image_path)
            
            # Log state to CSV.
            shape_name = shape.__class__.__name__.lower()
            csvwriter.writerow([
                frame, shape_name,
                shape.pos.x, shape.pos.y,
                shape.angle,
                shape.vel.x, shape.vel.y,
                shape.omega
            ])
            
# Exit pygame after running all iterations.
pygame.quit()