import numpy as np
import math

# Task 1.2 functions
def dot_product(v1,v2):
    dot_sum = 0
    for a, b in zip(v1,v2):
        dot_sum += a*b
    return dot_sum

def mag_sum(v1):
    mag_sum = 0
    for a in v1:
        mag_sum += a**2
    return math.sqrt(mag_sum)

def cosine_similarity_two(v1,v2):
    cos_similarity = (dot_product(v1,v2) / (mag_sum(v1)*mag_sum(v2)))
    return cos_similarity

# Task 1.2 validation across 100 random test cases
np.random.seed(42)
all_passed = True

for _ in range(100):
    # Generate random vectors
    n = np.random.randint(2, 100)
    v1 = np.random.rand(n).tolist()
    v2 = np.random.rand(n).tolist()
    
    # Manual
    manual_dot = dot_product(v1, v2)
    manual_mag1 = mag_sum(v1)
    manual_mag2 = mag_sum(v2)
    manual_cs = cosine_similarity_two(v1, v2)
    
    # Numpy
    numpy_dot = np.dot(v1, v2)
    numpy_mag1 = np.linalg.norm(v1)
    numpy_mag2 = np.linalg.norm(v2)
    numpy_cs = numpy_dot / (numpy_mag1 * numpy_mag2)
    
    # Compare
    if abs(manual_dot - numpy_dot) >= 1e-9:
        all_passed = False
        print(f"Failed on dot product: manual {manual_dot}, numpy {numpy_dot}")
        break
    if abs(manual_mag1 - numpy_mag1) >= 1e-9 or abs(manual_mag2 - numpy_mag2) >= 1e-9:
        all_passed = False
        print(f"Failed on magnitude: manual {manual_mag1}/{manual_mag2}, numpy {numpy_mag1}/{numpy_mag2}")
        break
    if abs(manual_cs - numpy_cs) >= 1e-9:
        all_passed = False
        print(f"Failed on cosine similarity: manual {manual_cs}, numpy {numpy_cs}")
        break

print(f"All 100 random test cases passed (difference < 1e-9)?: {all_passed}")