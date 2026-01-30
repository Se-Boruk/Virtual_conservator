import hashlib
import random

def Hash_generator(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Odczytujemy plik w blokach, aby nie przeciążyć RAMu przy dużych plikach
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

        
def get_random_from_hash(image_hash_string, min_val=0, max_val=22):
    seed_value = int(image_hash_string, 16)
    random.seed(seed_value)
    return random.randint(min_val, max_val)

def Clustering(file_path):
    hash = Hash_generator(file_path)
    return get_random_from_hash(hash)
