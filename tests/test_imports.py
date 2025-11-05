# test_imports.py
print("Testing discrete...")
import syntropy.discrete
print("✓ discrete OK")

print("Testing gaussian...")
import syntropy.gaussian  
print("✓ gaussian OK")

print("Testing knn...")
import syntropy.knn
print("✓ knn OK")

print("Testing neural...")
import syntropy.neural
print("✓ neural OK")

print("\n✓ All imports successful - no circular dependencies!")
