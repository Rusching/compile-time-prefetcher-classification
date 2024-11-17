import struct

trace_file = "1c.champsim"

# <:    little endian
# Q:    uint64  ip              8
# B:    uint8   is_branch       1
# B:    uint8   branch_taken    1
# 2B:   uint8   dest_regs       2
# 4B:   uint8   src_regs        4
# 2Q:   uint64  dest_mem        16
# 4Q:   uint64  src_mem         32

record_format = '<QBB2B4B2Q4Q'

# sum of 8 + 1 + 1 + 2 + 4 + 16 + 32
record_size = 64  

with open(trace_file, "rb") as f:
    while chunk := f.read(record_size):
        ip, is_branch, branch_taken, \
        dest_reg1, dest_reg2, \
        src_reg1, src_reg2, src_reg3, src_reg4, \
        dest_mem1, dest_mem2, \
        src_mem1, src_mem2, src_mem3, src_mem4 = struct.unpack(record_format, chunk)

        print(f"IP: {ip:#x}, Is Branch: {is_branch}, Branch Taken: {branch_taken}, Dest Regs: {dest_reg1}, {dest_reg2}, Src Regs: {src_reg1}, {src_reg2}, {src_reg3}, {src_reg4}, Dest Mem: {dest_mem1:#x}, {dest_mem2:#x}, Src Mem: {src_mem1:#x}, {src_mem2:#x}, {src_mem3:#x}, {src_mem4:#x}")
