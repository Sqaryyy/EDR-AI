import os

# maps ADFA syscall numbers to names (linux x86-64)
def syscall_name(num):
    mapping = {
        0: "read", 1: "write", 2: "open", 3: "close",
        4: "stat", 9: "mmap", 11: "munmap", 41: "socket",
        42: "connect", 56: "clone", 57: "fork", 59: "execve",
        82: "rename", 257: "openat", 260: "openat2", 316: "renameat2"
    }
    return mapping.get(num, "other")

def convert_file(src_path, dst_path, fake_pid=99999):
    with open(src_path) as f:
        content = f.read()
    
    import re
    numbers = re.findall(r'\d+', content)
    
    with open(dst_path, "w") as out:
        for n in numbers:
            try:
                num = int(n)
                out.write(f"{fake_pid},{num},{num}\n")  # use number as its own name
            except:
                continue

def convert_directory(src_dir, dst_dir, prefix):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(src_dir):
        src = os.path.join(src_dir, filename)
        if os.path.isfile(src):
            dst = os.path.join(dst_dir, f"{prefix}_{filename}.txt")
            convert_file(src, dst)
            count += 1
    print(f"Converted {count} files from {src_dir} -> {dst_dir}")

# convert normal training data
convert_directory(
    "ADFA-LD/ADFA-LD/Training_Data_Master",
    "logs/adfa_normal",
    "normal"
)

# convert attack data
attack_base = "ADFA-LD/ADFA-LD/Attack_Data_Master"
for attack_type in os.listdir(attack_base):
    src = os.path.join(attack_base, attack_type)
    if os.path.isdir(src):
        convert_directory(src, f"logs/adfa_attacks/{attack_type}", attack_type)

print("Done.")