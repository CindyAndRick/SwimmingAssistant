import argparse
datapath ='./data/'

def parse_args():
    parser = argparse.ArgumentParser(description="AI for swimming time series data classifcation")
    # ========================= Label Configs ==========================
    parser.add_argument("--label_num",type=int, default=5, help="ordinary swim")
    parser.add_argument("--ordinary", type=int, default=0, help="ordinary swim")
    parser.add_argument("--breath",   type=int, default=1, help="swim with breathing problems")
    parser.add_argument("--swing",    type=int, default=2, help="swing the body while swimming")
    parser.add_argument("--upp",      type=int, default=3, help="swim with upper body arise")
    parser.add_argument("--out",      type=int, default=4, help="swim with arms stretching out too much")
    # ========================= Data Configs ==========================
    parser.add_argument("--step_len", type=int, default=15)
    return parser.parse_args()