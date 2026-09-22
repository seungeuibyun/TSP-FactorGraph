import pandas as pd
from pyproj import Transformer

speed = pd.read_excel(
    "topis_speed.xlsx",
    dtype={"링크아이디": str}
)

speed_ids = set(
    speed["링크아이디"].str.strip()
)

vtx = pd.read_excel(
    r"C:\Users\guild\Dropbox\연구\코드\TSP_new_new\simul"
    r"\서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx",
    dtype={"LINK_ID": str}
)

vtx["LINK_ID"] = vtx["LINK_ID"].str.strip()

# 우리가 쓰는 183개 링크만
vtx = vtx[
    vtx["LINK_ID"].isin(speed_ids)
].copy()

# TOPIS 좌표는 meter 단위니까 여기서 1 km buffer
BUFFER = 1000

xmin = vtx["GRS80TM_X"].min() - BUFFER
xmax = vtx["GRS80TM_X"].max() + BUFFER

ymin = vtx["GRS80TM_Y"].min() - BUFFER
ymax = vtx["GRS80TM_Y"].max() + BUFFER

tf = Transformer.from_crs(
    "EPSG:5181",
    "EPSG:4326",
    always_xy=True
)

west, south = tf.transform(xmin, ymin)
east, north = tf.transform(xmax, ymax)

print("south =", south)
print("west  =", west)
print("north =", north)
print("east  =", east)

print("\nOverpass bbox:")
print(f"({south},{west},{north},{east})")