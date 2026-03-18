import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Từ điển ánh xạ (Mapping)
supplier_map = {
    'S': 'Yungchi', 'T': 'AKZO NOBEL', 'B': 'Beckers', 'C': 'Nan Pao',
    'U': 'Quali Poly', 'N': 'Nippon', 'K': 'Kansai', 'V': 'Valspar',
    'J': 'Valspar (Sherwin Williams)', 'L': 'KCC', 'R': 'Noroo', 'Q': 'Paoqun'
}

color_map = {
    '0': 'Clear', '1': 'Red', 'R': 'Red', 'O': 'Orange', '2': 'Orange',
    '3': 'Yellow', 'Y': 'Yellow', '4': 'Green', 'G': 'Green',
    '5': 'Blue', 'L': 'Blue', 'V': 'Violet', '6': 'Violet',
    'N': 'Brown', '7': 'Brown', 'T': 'White', 'H': 'White', 'W': 'White', '8': 'White',
    'A': 'Gray', 'C': 'Gray', '9': 'Gray', 'B': 'Black', 'S': 'Silver', 'M': 'Metallic'
}

# 2. Dữ liệu mẫu (Giả lập một tập dữ liệu lớn hơn một chút)
# Lưu ý: Ký tự thứ 7 (index 6) quyết định màu sắc.
data = {
    'Ma_Son': [
        'PJ6CD3WZS', # W -> White, J -> Valspar
        'PT2CD18ZS', # 8 -> White, T -> Akzo Nobel
        'PB5XY2TZS', # T -> White, B -> Beckers
        'PL5XY3WZS', # W -> White, L -> KCC
        'PJ2CD90ZS', # 0 -> Clear (sẽ bị loại khi lọc màu Trắng)
        'PT2CD1RZS'  # R -> Red (sẽ bị loại khi lọc màu Trắng)
    ],
    'Do_Bong': [85.2, 86.5, 84.1, 85.8, 95.0, 35.5]
}
df = pd.DataFrame(data)

# 3. Bóc tách dữ liệu từ Mã sơn
df['Nha_Cung_Cap'] = df['Ma_Son'].str[1].map(supplier_map)
df['Mau_Sac'] = df['Ma_Son'].str[6].map(color_map)

# ---------------------------------------------------------
# BƯỚC QUAN TRỌNG: LỌC THEO MÀU SẮC BẠN MUỐN PHÂN TÍCH
# ---------------------------------------------------------
mau_can_phan_tich = 'White'  # Bạn có thể thay đổi thành 'Red', 'Blue', 'Gray'...
df_filtered = df[df['Mau_Sac'] == mau_can_phan_tich].copy()

print(f"--- DỮ LIỆU ĐÃ LỌC CHO MÀU: {mau_can_phan_tich.upper()} ---")
print(df_filtered[['Ma_Son', 'Nha_Cung_Cap', 'Mau_Sac', 'Do_Bong']])

# 4. Thống kê độ bóng của màu này giữa các nhà cung cấp
bang_thong_ke = df_filtered.groupby('Nha_Cung_Cap')['Do_Bong'].agg(
    So_Luong=('Do_Bong', 'count'),
    Do_Bong_TB=('Do_Bong', 'mean'),
    Do_Lech_Chuan=('Do_Bong', 'std'),
    Min=('Do_Bong', 'min'),
    Max=('Do_Bong', 'max')
).reset_index()

print(f"\n--- THỐNG KÊ ĐỘ BÓNG MÀU {mau_can_phan_tich.upper()} THEO NHÀ CUNG CẤP ---")
print(bang_thong_ke)
