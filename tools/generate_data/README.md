# Bouncing Shapes Data Generator
This tool generates sequences of frames showing a shape bouncing around a square environment with a fairly accurate physics model. Shapes are affected by gravity, air resistance, friction, and have slightly inelastic collisions with the walls. Objects are simulated using pygame.

## Generating Data
Constants at the top of `generate.py` define how many sequences to generate and the number of frames in each sequence. Sequences can be generated using `python generate.py`, and are put in the output directory specified in the file. Once generated, `convert.py` is used to convert the image sequences to a NumPy data file. 