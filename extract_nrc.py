import pandas as pd
import sys
from pathlib import Path

EMO_COLS_CANON = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]

def main(xlsx_path: str, out_csv: str):
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        raise FileNotFoundError(xlsx)

    # 엑셀 읽기
    df = pd.read_excel(xlsx, engine="openpyxl")
    # 컬럼명 정규화
    norm_cols = {c: str(c).strip() for c in df.columns}
    df.rename(columns=norm_cols, inplace=True)

    # 영어 단어 컬럼 찾기
    word_col = None
    for cand in ["English (en)", "English", "word", "Word"]:
        if cand in df.columns:
            word_col = cand
            break
    if word_col is None:
        # 대소문자/공백 변형에 대비
        lowers = {c.lower(): c for c in df.columns}
        for cand in ["english (en)", "english", "word"]:
            if cand in lowers:
                word_col = lowers[cand]
                break
    if word_col is None:
        raise ValueError("영어 단어 컬럼(English (en)/English/word)을 찾지 못했습니다. 헤더를 확인하세요.")

    # 감정 컬럼 매핑(대소문자 무시)
    emo_map = {}
    lowers = {c.lower(): c for c in df.columns}
    for emo in EMO_COLS_CANON:
        if emo in lowers:
            emo_map[emo] = lowers[emo]
        else:
            # 일부 파일은 대문자/첫글자 대문자일 수 있음
            cand = emo.capitalize()     # Anger …
            if cand in df.columns:
                emo_map[emo] = cand
            else:
                cand = emo.upper()      # ANGER …
                if cand in df.columns:
                    emo_map[emo] = cand
                else:
                    raise ValueError(f"감정 컬럼 '{emo}' 를 찾지 못했습니다. 헤더를 확인하세요.")

    out = pd.DataFrame()
    out["word"] = df[word_col].astype(str).str.strip().str.lower()

    for emo in EMO_COLS_CANON:
        col = emo_map[emo]
        out[emo] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 빈 단어 제거 및 중복 단어 합산(안전)
    out = out[out["word"] != ""]
    out = out.groupby("word", as_index=False)[EMO_COLS_CANON].sum()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] saved: {out_csv}  rows={len(out)}  cols={len(out.columns)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_emolex_xlsx.py <input.xlsx> <output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
