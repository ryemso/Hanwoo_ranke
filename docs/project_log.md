# 한우 등급 예측 공모전 개인 기록 정리

## 1. 문제 정의

이번 프로젝트의 목표는 한우 개체의 최종 등급인 `LAST_GRADE`를 예측하는 것이었다. 예측 대상은 `1++A`, `1++B`, `1++C`, `1+A`, `1+B`, `1+C`, `1A`, `1B`, `1C`, `2A`, `2B`, `2C`, `3A`, `3B`, `3C`, `등외`까지 총 16개 클래스였다.

평가지표는 Macro-F1이었기 때문에 다수 클래스만 잘 맞히는 것보다 각 등급별 균형 성능이 중요했다.

## 2. 초기 접근과 누수 변수 확인

초기 train 데이터에는 도축 이후 판정 과정에서 확인되는 변수가 포함되어 있었다.

- `BACKFAT`
- `REA`
- `WINDEX`
- `WGRADE`
- `INSFAT`
- `YUKSAK`
- `FATSAK`
- `TISSUE`
- `GROWTH`
- `COST_AMT`

특히 `WGRADE`, `WINDEX` 등은 최종 등급과 직접적으로 연결되는 변수였기 때문에 내부 검증에서는 높은 성능을 만들 수 있었다. 하지만 test 데이터에는 해당 변수들이 없었기 때문에 실제 제출에는 사용할 수 없는 누수성 변수로 판단했다.

이후 방향을 test 가용 변수 중심으로 재설계했다.

## 3. 사용 가능한 변수 기준 재설계

test에서 활용 가능한 정보는 다음과 같았다.

- 지역 정보: `sido`, `sigungu`, `eupmyeondong`, `stn`
- 날짜 정보: `ABATT_DATE`, `JUDGE_DATE`, `BIRTH_YMD`
- 개체 정보: `CATTLE_NO`, `JUDGE_SEX`, `WEIGHT`, `AGE`
- 농장 정보: `FARM_UNIQUE_NO`
- 외부 결합 정보: 지역/농장 데이터, 날씨 데이터, 혈통 데이터, KPN 유전능력 자료, 폐사 통계

이 구조에서는 도축 후 판정 변수 없이 최종 등급을 간접 추정해야 했기 때문에, 단순 모델링보다 피처 엔지니어링이 중요했다.

## 4. 데이터 결합

### 4.1 지역/농장 데이터

`hanwoo_area.csv`를 활용해 농장 수, 면적, 농장 밀도 등의 파생 변수를 만들었다.

### 4.2 혈통 데이터

`hanwoo_lineage_0612.csv`를 `CATTLE_NO` 기준으로 결합했다. 주요 활용 변수는 다음과 같다.

- `KPN_NO`
- `FATHER_CATTLE_NO`
- `MOTHER_ANIMAL_NO`
- 조부모 계통 정보

### 4.3 KPN 유전능력 자료

`KPN 유전능력 자료.xlsx`를 `KPN_NO` 기준으로 결합하여 혈통 기반 유전능력 정보를 반영했다.

### 4.4 기상 데이터

`hanwoo_weather.csv`를 관측지점과 날짜 기준으로 결합했다. 활용 변수는 다음과 같다.

- 최고기온
- 최저기온
- 강수량
- 평균습도
- 평균풍속

## 5. 주요 피처 엔지니어링

### 5.1 날짜 파생

출생일, 도축일, 판정일을 datetime으로 변환하고 연, 월, 계절, 경과일 관련 변수를 만들었다.

### 5.2 체중·나이 파생

`WEIGHT`, `AGE`를 기반으로 체중-나이 상호작용을 만들었다. 회고 관점에서는 `weight_per_age`, 성별/지역/농장/KPN 대비 상대 체중 z-score를 더 강하게 만들었으면 좋았을 것으로 판단했다.

### 5.3 고카디널리티 범주형 인코딩

다음 변수들은 빈도 인코딩을 적용했다.

- 지역 변수
- 농장번호
- KPN 번호
- 부모/조부모 계통
- 지역+성별 조합
- 농장+성별 조합
- KPN+성별 조합

### 5.4 rolling history feature

가장 중요한 피처 중 하나는 과거 이력 feature였다. 농장, 지역, 혈통 계열이 과거에 어떤 등급을 배출했는지를 반영했다.

사용한 주요 target metric은 다음과 같다.

- `grade_score`
- `is_1pp`
- `is_1plus_or_up`
- `is_A`
- `is_B`
- `is_C`
- `is_out`

주요 key는 다음과 같다.

- `FARM_UNIQUE_NO`
- `farm_sex`
- `region_key`
- `region_sex_key`
- `KPN_NO`
- `kpn_sex`

train 검증에서는 시간 순서를 고려한 causal rolling을 사용했고, test 예측에서는 전체 train 이력을 사용하는 full history를 구성했다.

## 6. 모델링

최종 모델은 CatBoost와 XGBoost를 중심으로 구성했다.

- CatBoost: 범주형 패턴과 비선형 관계 학습
- XGBoost: 수치형 파생 변수 간 상호작용 학습

최종 예측 확률은 다음 방식으로 결합했다.

```text
final_proba = 0.6 * CatBoost_proba + 0.4 * XGBoost_proba
```

이후 temperature scaling을 적용해 확률 분포를 완화했다.

## 7. 실험 흐름

초기 base 모델은 public Macro-F1 약 0.210 수준이었다.

| 실험 | Public Macro-F1 |
|---|---:|
| zscore Cat+XGB base fullhist | 0.210 |
| base0640 notemp fullhist | 0.210 |
| base0730 fullhist | 0.210 |
| cat only fullhist | 0.207 |
| xgb only fullhist | 0.202 |

이후 seed와 class weight alpha를 조정했다.

| 실험 | Public Macro-F1 |
|---|---:|
| seed2026 단독 | 0.200 |
| alpha0.35 seed777 단독 | 0.202 |
| seed42 + alpha0.35 seed777 평균 | 0.214 |

약한 단일 모델이라도 오류 패턴이 다르면 앙상블에서 보완 효과가 있음을 확인했다.

## 8. 500k 모델과 C계열 문제

500k 샘플 기반 모델은 validation에서는 Macro-F1 약 0.239 수준이었지만 public에서는 0.212에 그쳤다.

이 과정에서 C계열 등급을 실제보다 적게 예측하는 문제가 확인되었다. C계열은 도체 수율, 등지방, 등심단면적 등 test에 없는 직접 변수와 밀접하게 연결되어 있어 간접 추정이 어려웠다.

## 9. 1M 모델과 체크포인트 관리

마지막 단계에서 Colab 유료 환경을 활용해 1M 샘플 기반 모델을 학습했다. 학습 도중 실행이 중단되는 문제가 있었지만, `prep` 객체가 살아 있어 전처리 결과를 체크포인트로 저장했다.

저장한 주요 파일은 다음과 같다.

- `X_train_imp.npy`
- `X_valid_imp.npy`
- `y_train_enc.npy`
- `y_valid_enc.npy`
- `sample_weight_large.npy`
- `y_train.pkl`
- `y_valid.pkl`
- `imputer.pkl`
- `zscore_stats.pkl`
- `num_cols.pkl`
- `feature_names_final.pkl`
- `freq_maps.pkl`

이후 CatBoost와 XGBoost를 각각 따로 학습하고 즉시 저장하는 방식으로 변경했다.

## 10. C계열 확률 보정

최종 병목이 C계열 과소예측이라고 판단하여 확률 기반 후처리를 적용했다. hard label correction은 사용하지 않고, 예측 확률에 multiplier를 적용한 뒤 정규화했다.

대표 보정값은 다음과 같다.

```text
품질축:
1++ = 0.98
1+  = 1.03
1   = 1.02
2   = 1.00
3   = 1.00
등외 = 1.00

suffix축:
A = 0.98
B = 1.00
C = 1.15
등외 = 1.00
```

## 11. 최종 결과

| 실험 | Public Macro-F1 |
|---|---:|
| 기존 최고 seed ensemble | 0.214 |
| best80 + colab1m adjusted20 | 0.218 |
| colab1m cboost115 adjusted 단독 | 0.219 |
| colab1m cboost118 adjusted 단독 | 0.219 |

최종 최고 public Macro-F1은 0.219였다.

## 12. 아쉬운 점

프로젝트 종료 후 회고해보면, 단순 파생컬럼보다 등급 구조를 겨냥한 파생컬럼이 더 필요했다.

추가했으면 좋았을 feature는 다음과 같다.

- `weight_per_age`
- 성별/지역/농장/KPN 대비 상대 체중 z-score
- 품질축 score history
- 수율축 A/B/C score history
- C-risk feature
- 출하 전 누적 기상 스트레스
- season × region × sex interaction
- 품질축 모델과 A/B/C suffix 모델을 분리한 multi-stage 모델

## 13. 배운 점

한우 등급 예측은 단순한 분류 문제가 아니라 도메인 결정 구조를 이해해야 하는 문제였다. 특히 train에는 존재하지만 test에는 없는 도축 후 판정 변수를 제거하고, 실제 예측 가능한 변수만으로 모델을 재설계하는 과정이 중요했다.

또한 모델 성능을 올리는 것만큼이나 제출 파일 관리, checkpoint 저장, 마지막 제출 파일 고정 같은 운영 판단도 중요하다는 점을 배웠다.
