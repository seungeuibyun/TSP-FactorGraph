import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

speed = pd.read_excel(r"C:\Users\guild\Dropbox\연구\코드\TSP_new_new\simul\2026년 5월 서울시 차량통행속도.xlsx", dtype={"링크아이디": str})
vertex = pd.read_excel(
    "서비스링크 보간점 정보(LINK_VERTEX)_2025.xlsx",
    dtype={"LINK_ID": str}
)

# 네 속도 데이터에 실제 존재하는 링크만 추림
vertex = vertex[
    vertex["LINK_ID"].isin(speed["링크아이디"])
].copy()

# 링크별 vertex 순서 정렬
vertex = vertex.sort_values(["LINK_ID", "VER_SEQ"])

lines = (
    vertex.groupby("LINK_ID")
    .apply(
        lambda x: LineString(
            zip(x["GRS80TM_X"], x["GRS80TM_Y"])
        )
    )
)

links_gdf = gpd.GeoDataFrame(
    {"LINK_ID": lines.index},
    geometry=lines.values,
    crs="EPSG:5181"   # 좌표계는 파일 명세 확인 후 확정
)

speed_ids = set(speed["링크아이디"].astype(str))
vertex_ids = set(vertex["LINK_ID"].astype(str))

matched = speed_ids & vertex_ids

print(len(speed_ids))
print(len(matched))
print(len(matched) / len(speed_ids))
