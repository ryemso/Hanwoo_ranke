"""
Hanwoo Grade Prediction Pipeline
================================

공모전에서 사용한 한우 등급 예측 모델링 흐름을 재현 가능한 형태로 정리한 코드입니다.
원천 데이터와 모델 산출물은 저장소에 포함하지 않습니다.

주요 흐름
- 데이터 로드
- 결측/이상치 처리
- 지역/혈통/KPN/기상 데이터 결합
- 날짜/체중/나이/범주형 빈도/rolling history feature 생성
- CatBoost + XGBoost 학습
- 확률 앙상블 및 C계열 확률 보정
- 제출 파일 생성
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

from catboost import CatBoostClassifier
from xgboost import XGBClassifier


GRADE_ORDER = [
    "1++A", "1++B", "1++C",
    "1+A", "1+B", "1+C",
    "1A", "1B", "1C",
    "2A", "2B", "2C",
    "3A", "3B", "3C",
    "등외",
]

GRADE_ARR = np.array(GRADE_ORDER)
TARGET = "LAST_GRADE"


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """공모전 데이터 내 결측 표시값을 np.nan으로 통일합니다."""
    return df.replace(
        [-99, -99.0, "-99", "-99.0", "", " ", "nan", "NaN", "NULL"],
        np.nan,
    )


def reduce_numeric_memory(df: pd.DataFrame) -> pd.DataFrame:
    """대용량 데이터 처리를 위해 수치형 dtype을 downcast합니다."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def safe_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def get_quality_suffix(label: str):
    """LAST_GRADE를 품질축과 A/B/C suffix 축으로 분해합니다."""
    if label == "등외":
        return "등외", "등외"
    if label.startswith("1++"):
        return "1++", label.replace("1++", "")
    if label.startswith("1+"):
        return "1+", label.replace("1+", "")
    return label[0], label[1:]


QUALITY_OF_GRADE = {}
SUFFIX_OF_GRADE = {}
for _g in GRADE_ORDER:
    _q, _s = get_quality_suffix(_g)
    QUALITY_OF_GRADE[_g] = _q
    SUFFIX_OF_GRADE[_g] = _s


def grade_to_score(label: str) -> float:
    """등급을 history feature용 연속 점수로 변환합니다."""
    mapping = {
        "1++A": 15, "1++B": 14, "1++C": 13,
        "1+A": 12, "1+B": 11, "1+C": 10,
        "1A": 9, "1B": 8, "1C": 7,
        "2A": 6, "2B": 5, "2C": 4,
        "3A": 3, "3B": 2, "3C": 1,
        "등외": 0,
    }
    return mapping.get(str(label), np.nan)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["ABATT_DATE", "JUDGE_DATE", "BIRTH_YMD"]:
        if col in df.columns:
            df[col] = safe_datetime(df[col])

    if "ABATT_DATE" in df.columns:
        df["abatt_year"] = df["ABATT_DATE"].dt.year
        df["abatt_month"] = df["ABATT_DATE"].dt.month
        df["abatt_day"] = df["ABATT_DATE"].dt.day
        df["abatt_dow"] = df["ABATT_DATE"].dt.dayofweek
        df["abatt_quarter"] = df["ABATT_DATE"].dt.quarter
        df["abatt_season"] = ((df["abatt_month"] % 12) // 3 + 1).astype("float")

    if "JUDGE_DATE" in df.columns:
        df["judge_year"] = df["JUDGE_DATE"].dt.year
        df["judge_month"] = df["JUDGE_DATE"].dt.month
        df["judge_dow"] = df["JUDGE_DATE"].dt.dayofweek

    if "ABATT_DATE" in df.columns and "BIRTH_YMD" in df.columns:
        df["birth_to_abatt_days"] = (df["ABATT_DATE"] - df["BIRTH_YMD"]).dt.days
        df["birth_to_abatt_months"] = df["birth_to_abatt_days"] / 30.4375

    if "JUDGE_DATE" in df.columns and "ABATT_DATE" in df.columns:
        df["judge_after_abatt_days"] = (df["JUDGE_DATE"] - df["ABATT_DATE"]).dt.days

    return df


def add_weight_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["WEIGHT", "AGE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "WEIGHT" in df.columns and "AGE" in df.columns:
        df["weight_per_age"] = df["WEIGHT"] / np.clip(df["AGE"], 1, None)
        df["weight_x_age"] = df["WEIGHT"] * df["AGE"]
        df["weight_div_age2"] = df["WEIGHT"] / np.clip(df["AGE"] ** 2, 1, None)
        df["age_bin"] = pd.cut(
            df["AGE"],
            bins=[0, 24, 28, 32, 36, 42, 1000],
            labels=False,
            include_lowest=True,
        )
        df["weight_bin"] = pd.cut(
            df["WEIGHT"],
            bins=[0, 500, 600, 650, 700, 750, 800, 900, 2000],
            labels=False,
            include_lowest=True,
        )

    return df


def build_region_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["sido", "sigungu", "eupmyeondong", "stn", "FARM_UNIQUE_NO", "KPN_NO"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    if {"sido", "sigungu"}.issubset(df.columns):
        df["region_key"] = df["sido"].fillna("NA") + "_" + df["sigungu"].fillna("NA")

    if {"region_key", "JUDGE_SEX"}.issubset(df.columns):
        df["region_sex_key"] = df["region_key"].astype("string") + "_" + df["JUDGE_SEX"].astype("string")

    if {"FARM_UNIQUE_NO", "JUDGE_SEX"}.issubset(df.columns):
        df["farm_sex"] = df["FARM_UNIQUE_NO"].astype("string") + "_" + df["JUDGE_SEX"].astype("string")

    if {"KPN_NO", "JUDGE_SEX"}.issubset(df.columns):
        df["kpn_sex"] = df["KPN_NO"].astype("string") + "_" + df["JUDGE_SEX"].astype("string")

    if {"sigungu", "JUDGE_SEX"}.issubset(df.columns):
        df["sigungu_sex"] = df["sigungu"].astype("string") + "_" + df["JUDGE_SEX"].astype("string")

    if {"eupmyeondong", "JUDGE_SEX"}.issubset(df.columns):
        df["eup_sex"] = df["eupmyeondong"].astype("string") + "_" + df["JUDGE_SEX"].astype("string")

    return df


def frequency_encode_fit(train_df: pd.DataFrame, cols):
    maps = {}
    n = len(train_df)
    for col in cols:
        if col in train_df.columns:
            vc = train_df[col].astype("string").fillna("NA").value_counts(dropna=False)
            maps[col] = (vc / max(n, 1)).to_dict()
    return maps


def frequency_encode_transform(df: pd.DataFrame, maps):
    df = df.copy()
    for col, mp in maps.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].astype("string").fillna("NA").map(mp).fillna(0).astype("float32")
    return df


def make_common_features(df, area=None, lineage=None, kpn=None, weather=None, death_agg=None):
    """train/test 공통 feature 생성 함수."""
    df = clean_missing_values(df.copy())

    if area is not None and "FARM_UNIQUE_NO" in df.columns and "FARM_UNIQUE_NO" in area.columns:
        df = df.merge(area, on="FARM_UNIQUE_NO", how="left")

    if lineage is not None and "CATTLE_NO" in df.columns and "CATTLE_NO" in lineage.columns:
        df = df.merge(lineage, on="CATTLE_NO", how="left")

    if kpn is not None and "KPN_NO" in df.columns and "KPN_NO" in kpn.columns:
        df = df.merge(kpn, on="KPN_NO", how="left")

    if weather is not None and {"stn", "ABATT_DATE"}.issubset(df.columns):
        w = weather.copy()
        if "date" in w.columns:
            w["date"] = safe_datetime(w["date"])
            df["ABATT_DATE"] = safe_datetime(df["ABATT_DATE"])
            df = df.merge(w, left_on=["stn", "ABATT_DATE"], right_on=["stn", "date"], how="left")

    if death_agg is not None:
        merge_keys = [c for c in ["sido", "sigungu", "eupmyeondong", "stn"] if c in df.columns and c in death_agg.columns]
        if merge_keys:
            df = df.merge(death_agg, on=merge_keys, how="left", suffixes=("", "_death"))

    df = add_date_features(df)
    df = add_weight_age_features(df)
    df = build_region_keys(df)

    if {"C2023", "C2024", "C2025", "AREA"}.issubset(df.columns):
        df["farm_count_mean"] = df[["C2023", "C2024", "C2025"]].mean(axis=1)
        df["farm_density"] = df["farm_count_mean"] / np.clip(df["AREA"], 1, None)

    if {"WEIGHT", "AGE", "JUDGE_SEX"}.issubset(df.columns):
        df["sex_weight_key"] = df["JUDGE_SEX"].astype("string") + "_" + df["weight_bin"].astype("string")
        df["sex_age_key"] = df["JUDGE_SEX"].astype("string") + "_" + df["age_bin"].astype("string")

    return reduce_numeric_memory(df)


def add_target_auxiliary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grade_score"] = df[TARGET].map(grade_to_score).astype("float32")
    df["is_1pp"] = df[TARGET].astype(str).str.startswith("1++").astype("float32")
    df["is_1plus_or_up"] = df[TARGET].astype(str).str.startswith(("1++", "1+")).astype("float32")
    df["is_A"] = df[TARGET].astype(str).str.endswith("A").astype("float32")
    df["is_B"] = df[TARGET].astype(str).str.endswith("B").astype("float32")
    df["is_C"] = df[TARGET].astype(str).str.endswith("C").astype("float32")
    df["is_out"] = (df[TARGET].astype(str) == "등외").astype("float32")
    return df


def add_causal_rolling_features(train_like_df, keys_windows=None, metrics=None, date_col="ABATT_DATE"):
    """시간 순서를 보존한 causal rolling feature를 생성합니다."""
    if keys_windows is None:
        keys_windows = {
            "FARM_UNIQUE_NO": [10, 30],
            "farm_sex": [10, 30],
            "region_key": [30, 100],
            "region_sex_key": [30, 100],
            "KPN_NO": [20, 50],
            "kpn_sex": [20, 50],
        }
    if metrics is None:
        metrics = ["grade_score", "is_1pp", "is_1plus_or_up", "is_A", "is_B", "is_C", "is_out"]

    df = add_target_auxiliary_columns(train_like_df)
    df = df.sort_values(date_col).reset_index(drop=True)

    for key, windows in keys_windows.items():
        if key not in df.columns:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            grouped = df.groupby(key, dropna=False)[metric]
            for w in windows:
                name = f"hist_{key}_{metric}_w{w}"
                df[name] = grouped.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    return reduce_numeric_memory(df)


def build_history_context_light(train_all):
    """test full history 변환을 위한 key별 평균 통계를 생성합니다."""
    train_all = add_target_auxiliary_columns(train_all)
    ctx = {}

    keys = ["FARM_UNIQUE_NO", "farm_sex", "region_key", "region_sex_key", "KPN_NO", "kpn_sex"]
    metrics = ["grade_score", "is_1pp", "is_1plus_or_up", "is_A", "is_B", "is_C", "is_out"]

    for key in keys:
        if key not in train_all.columns:
            continue
        agg = train_all.groupby(key, dropna=False)[metrics].mean().reset_index()
        ctx[key] = agg

    global_means = train_all[metrics].mean().to_dict()
    ctx["__global__"] = global_means
    return ctx


def transform_rolling_features(df, hist_ctx):
    """test 데이터에 full history feature를 결합합니다."""
    df = df.copy()
    global_means = hist_ctx.get("__global__", {})

    for key, agg in hist_ctx.items():
        if key == "__global__" or key not in df.columns:
            continue
        rename_map = {c: f"hist_{key}_{c}_full" for c in agg.columns if c != key}
        tmp = agg.rename(columns=rename_map)
        df = df.merge(tmp, on=key, how="left")
        for new_col in rename_map.values():
            base_metric = new_col.replace(f"hist_{key}_", "").replace("_full", "")
            if new_col in df.columns:
                df[new_col] = df[new_col].fillna(global_means.get(base_metric, 0))

    return reduce_numeric_memory(df)


def fit_zscore_bounds(df, num_cols, z=5.0):
    stats = {}
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            mu = s.mean()
            sd = s.std()
            if pd.notna(sd) and sd > 0:
                stats[c] = (mu - z * sd, mu + z * sd)
    return stats


def apply_zscore_clipping(df, stats):
    df = df.copy()
    for c, (lo, hi) in stats.items():
        if c in df.columns:
            val = pd.to_numeric(df[c], errors="coerce")
            df[f"{c}_zout"] = ((val < lo) | (val > hi)).astype("int8")
            df[c] = val.clip(lo, hi)
    return df


def make_sample_weight(y_enc, alpha=0.5):
    classes = np.unique(y_enc)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_enc)
    weight_map = {c: w for c, w in zip(classes, weights)}
    return np.array([weight_map[y] for y in y_enc], dtype="float32") ** alpha


def temperature_scale_proba(p, temp=1.1):
    p = np.clip(p, 1e-9, 1.0)
    logp = np.log(p) / temp
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out.astype("float32")


def align_encoded_grade_proba(raw_proba, classes_):
    out = np.zeros((raw_proba.shape[0], len(GRADE_ORDER)), dtype="float32")
    for j, cls in enumerate(classes_):
        idx = int(cls)
        if 0 <= idx < len(GRADE_ORDER):
            out[:, idx] = raw_proba[:, j]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out


def adjust_grade_proba(proba, quality_mult=None, suffix_mult=None, class_mult=None):
    """품질축/suffix축/class별 multiplier로 예측 확률을 보정합니다."""
    p = proba.copy().astype("float32")
    mult = np.ones(len(GRADE_ORDER), dtype="float32")

    if quality_mult is not None:
        for i, g in enumerate(GRADE_ORDER):
            mult[i] *= quality_mult.get(QUALITY_OF_GRADE[g], 1.0)

    if suffix_mult is not None:
        for i, g in enumerate(GRADE_ORDER):
            mult[i] *= suffix_mult.get(SUFFIX_OF_GRADE[g], 1.0)

    if class_mult is not None:
        for i, g in enumerate(GRADE_ORDER):
            mult[i] *= class_mult.get(g, 1.0)

    p *= mult.reshape(1, -1)
    p /= np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
    return p.astype("float32")


def train_cat_xgb(X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, random_state=777):
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=900,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=2.0,
        bagging_temperature=1.0,
        border_count=128,
        bootstrap_type="Bayesian",
        random_seed=random_state,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )

    cat.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=(X_valid, y_valid) if X_valid is not None else None,
        early_stopping_rounds=100 if X_valid is not None else None,
        use_best_model=True if X_valid is not None else False,
        verbose=100,
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(GRADE_ORDER),
        n_estimators=900,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        reg_alpha=0.5,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    xgb.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)] if X_valid is not None else None,
        verbose=100,
    )

    return cat, xgb


def predict_cat_xgb_proba(cat, xgb, X, w_cat=0.6, w_xgb=0.4, temp=1.1):
    cat_raw = cat.predict_proba(X).astype("float32")
    xgb_raw = xgb.predict_proba(X).astype("float32")

    cat_p = align_encoded_grade_proba(cat_raw, getattr(cat, "classes_", np.arange(len(GRADE_ORDER))))
    xgb_p = align_encoded_grade_proba(xgb_raw, getattr(xgb, "classes_", np.arange(len(GRADE_ORDER))))

    cat_t = temperature_scale_proba(cat_p, temp)
    xgb_t = temperature_scale_proba(xgb_p, temp)

    p = w_cat * cat_t + w_xgb * xgb_t
    p /= np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
    return p.astype("float32")


def save_model_pack(save_dir, cat, xgb, imputer, zscore_stats, num_cols, feature_names_final, freq_maps):
    os.makedirs(save_dir, exist_ok=True)
    cat.save_model(os.path.join(save_dir, "cat_model.cbm"))
    xgb.save_model(os.path.join(save_dir, "xgb_model.json"))
    joblib.dump(imputer, os.path.join(save_dir, "imputer.pkl"))
    joblib.dump(zscore_stats, os.path.join(save_dir, "zscore_stats.pkl"))
    joblib.dump(num_cols, os.path.join(save_dir, "num_cols.pkl"))
    joblib.dump(feature_names_final, os.path.join(save_dir, "feature_names_final.pkl"))
    joblib.dump(freq_maps, os.path.join(save_dir, "freq_maps.pkl"))


def load_model_pack(save_dir):
    cat = CatBoostClassifier()
    cat.load_model(os.path.join(save_dir, "cat_model.cbm"))

    xgb = XGBClassifier()
    xgb.load_model(os.path.join(save_dir, "xgb_model.json"))

    return {
        "cat": cat,
        "xgb": xgb,
        "imputer": joblib.load(os.path.join(save_dir, "imputer.pkl")),
        "zscore_stats": joblib.load(os.path.join(save_dir, "zscore_stats.pkl")),
        "num_cols": joblib.load(os.path.join(save_dir, "num_cols.pkl")),
        "feature_names_final": joblib.load(os.path.join(save_dir, "feature_names_final.pkl")),
        "freq_maps": joblib.load(os.path.join(save_dir, "freq_maps.pkl")),
    }


def prepare_test_features(test_df, pack, area=None, lineage=None, kpn=None, weather=None, death_agg=None, hist_ctx_all=None):
    df = make_common_features(test_df, area=area, lineage=lineage, kpn=kpn, weather=weather, death_agg=death_agg)
    df = frequency_encode_transform(df, pack["freq_maps"])
    if hist_ctx_all is not None:
        df = transform_rolling_features(df, hist_ctx_all)
    df = apply_zscore_clipping(df, pack["zscore_stats"])

    X = df.reindex(columns=pack["feature_names_final"])
    X_imp = pack["imputer"].transform(X)
    return X_imp


def predict_pack_proba(pack, X_imp, w_cat=0.6, w_xgb=0.4, temp=1.1):
    return predict_cat_xgb_proba(pack["cat"], pack["xgb"], X_imp, w_cat=w_cat, w_xgb=w_xgb, temp=temp)


def make_submission_from_pack(
    test_path,
    output_path,
    pack,
    area=None,
    lineage=None,
    kpn=None,
    weather=None,
    death_agg=None,
    hist_ctx_all=None,
    adjust_cfg=None,
    chunksize=50000,
):
    first = True
    total = 0

    for test_chunk in pd.read_csv(test_path, chunksize=chunksize, na_values=[-99, "-99", "-99.0"]):
        out = test_chunk.copy()
        X_imp = prepare_test_features(
            test_chunk,
            pack,
            area=area,
            lineage=lineage,
            kpn=kpn,
            weather=weather,
            death_agg=death_agg,
            hist_ctx_all=hist_ctx_all,
        )
        p = predict_pack_proba(pack, X_imp)

        if adjust_cfg is not None:
            p = adjust_grade_proba(
                p,
                quality_mult=adjust_cfg.get("quality_mult"),
                suffix_mult=adjust_cfg.get("suffix_mult"),
                class_mult=adjust_cfg.get("class_mult"),
            )

        out[TARGET] = GRADE_ARR[np.argmax(p, axis=1)]
        out.to_csv(output_path, mode="w" if first else "a", header=first, index=False, encoding="utf-8-sig")
        first = False
        total += len(out)
        print("processed:", total)

        del out, X_imp, p, test_chunk
        gc.collect()

    print("saved:", output_path)
    return output_path


def default_cboost115_cfg():
    return {
        "quality_mult": {
            "1++": 0.98,
            "1+": 1.03,
            "1": 1.02,
            "2": 1.00,
            "3": 1.00,
            "등외": 1.00,
        },
        "suffix_mult": {
            "A": 0.98,
            "B": 1.00,
            "C": 1.15,
            "등외": 1.00,
        },
        "class_mult": None,
    }


def check_submission(path, grade_order=GRADE_ORDER):
    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    if TARGET in df.columns:
        print(df[TARGET].value_counts().reindex(grade_order).fillna(0).astype(int))
        invalid = set(df[TARGET].astype(str).unique()) - set(grade_order)
        print("invalid labels:", invalid)
    return df


if __name__ == "__main__":
    print("Hanwoo pipeline module loaded.")
    print("Set data paths and call the functions in notebook/Colab for full execution.")
