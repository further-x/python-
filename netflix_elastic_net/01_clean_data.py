"""第一步：清洗 Netflix 电影数据。运行：python 01_clean_data.py"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / 'netflix_titles.csv')
movies = raw.loc[raw['type'].eq('Movie')].copy()

# 三条电影记录的时长被放进了 rating；只修复 duration 缺失的明确情况。
misplaced = movies['duration'].isna() & movies['rating'].str.fullmatch(r'\d+ min', na=False)
movies.loc[misplaced, 'duration'] = movies.loc[misplaced, 'rating']
movies.loc[misplaced, 'rating'] = pd.NA

# 只接受明确以分钟为单位的时长；不把电视剧季数当作分钟。
movies['duration_min'] = pd.to_numeric(
    movies['duration'].str.extract(r'^(\d+) min$')[0], errors='coerce')
valid = movies['duration_min'].notna() & movies['duration_min'].gt(0)
clean = movies.loc[valid].copy()
clean['duration_min'] = clean['duration_min'].astype(int)

# 缺失的类别显式标为 Unknown；不删除缺少导演或演员的电影。
for col in ['country', 'rating', 'listed_in']:
    clean[col] = clean[col].fillna('Unknown').str.strip().replace('', 'Unknown')

# 国家、类型保留完整列表，后续建模时再按逗号拆分并编码。
# 不在此阶段对全数据做标准化或学习类别，避免测试数据泄漏。
clean.to_csv(ROOT / 'movies_clean.csv', index=False, encoding='utf-8-sig')
summary = {
    'original_rows': len(raw), 'original_movies': len(movies),
    'repaired_duration_rows': int(misplaced.sum()),
    'dropped_invalid_duration_rows': int((~valid).sum()),
    'clean_movies': len(clean),
    'duration_statistics': clean['duration_min'].describe().to_dict(),
    'unknown_country_rows': int(clean['country'].eq('Unknown').sum()),
    'unknown_rating_rows': int(clean['rating'].eq('Unknown').sum()),
}
(ROOT / 'cleaning_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
