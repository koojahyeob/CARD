import pandas as pd

def load_and_basic_clean(path, time_col):
    """
    데이터 로드 및 기본 정리
    """
    # 데이터 로드
    df = pd.read_csv(path, encoding="utf-8")

    # 디버깅: 데이터셋의 컬럼 확인
    print("Columns in dataset:", df.columns.tolist())

    # 문자열 → datetime64[ns]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except KeyError as e:
        raise KeyError(f"Column '{time_col}' not found in dataset. Available columns: {df.columns.tolist()}")

    # 시간순 정렬
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def preprocess_for_card(input_path, output_path, time_col="ppltn_time", region_col="area_nm", target_cols=None):
    """
    데이터셋을 CARD 모델 입력 형식으로 변환하여 저장합니다.
    """
    if target_cols is None:
        target_cols = ["ppltn_rate0", "ppltn_rate10", "ppltn_rate20", "ppltn_rate30",
                       "ppltn_rate40", "ppltn_rate50", "ppltn_rate60", "ppltn_rate70"]

    # 데이터 로드
    df = pd.read_csv(input_path, encoding="utf-8")

    # 중복 확인 및 처리
    duplicates = df.duplicated(subset=[time_col, region_col], keep=False)
    if duplicates.any():
        print("중복된 항목이 발견되었습니다. 중복 데이터를 처리합니다.")
        # 중복 데이터를 평균으로 집계
        df = df.groupby([time_col, region_col], as_index=False).mean()

    # 필요한 컬럼만 선택
    selected_cols = [time_col, region_col] + target_cols
    try:
        df = df[selected_cols]
    except KeyError as e:
        raise KeyError(f"Missing columns in dataset: {e}")

    # 지역별로 데이터를 피벗
    try:
        pivoted = df.pivot(index=time_col, columns=region_col, values=target_cols)
    except ValueError as e:
        raise ValueError(f"Pivot operation failed: {e}")

    # 컬럼 이름을 "지역이름_컬럼명" 형식으로 변경
    pivoted.columns = [f"{region}__{col}" for region, col in pivoted.columns]

    # 인덱스를 리셋하여 시간 컬럼 포함
    pivoted.reset_index(inplace=True)

    # 결과 저장
    pivoted.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    return pivoted