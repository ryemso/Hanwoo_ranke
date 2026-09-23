# Hanwoo Grade Prediction

한우 개체 정보, 혈통 관계, 농장/지역 이력, 지역별 기상 데이터를 융합하여 `LAST_GRADE` 16개 등급을 예측한 공모전 프로젝트입니다.

## 프로젝트 요약

- **목표**: `1++A`부터 `등외`까지 총 16개 한우 최종 등급 다중 분류
- **평가지표**: Macro-F1
- **핵심 데이터**: 개체 정보, 도축/판정일, 체중, 나이, 성별, 농장 정보, 지역 정보, 혈통/KPN, 기상 데이터
- **모델**: CatBoost + XGBoost 확률 앙상블
- **핵심 전략**: test에서 사용 가능한 변수만 활용, 농장/지역/혈통 단위 history feature, C계열 과소예측 보정
- **최종 Public Macro-F1**: 0.219

## 프로젝트 구조

```text
Hanwoo_ranke/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── hanwoo_pipeline.py
└── docs/
    ├── project_log.md
    └── report_summary.md
```

## 주요 문제 인식

초기 train 데이터에는 `BACKFAT`, `REA`, `WINDEX`, `WGRADE`, `INSFAT`, `YUKSAK`, `FATSAK`, `TISSUE`, `GROWTH`, `COST_AMT` 등 최종 등급과 직접 연결되는 도축 후 판정 변수가 존재했습니다. 하지만 test 데이터에는 해당 변수들이 없었기 때문에, 이를 사용하는 모델은 내부 검증 성능은 높아도 실제 제출에는 사용할 수 없는 누수 구조였습니다.

따라서 본 프로젝트는 실제 test에서 사용 가능한 변수만 기준으로 재설계했습니다.

## 사용 데이터

- 한우 개체 기본 데이터
- 한우 지역/농장 면적 및 농장 수 데이터
- 한우 혈통 데이터
- KPN 유전능력 자료
- 지역별 기상 데이터
- 폐사 관련 통계 데이터

대용량 원천 데이터와 제출 CSV는 GitHub에 포함하지 않았습니다. 실행 시 `base_path` 아래에 원본 데이터를 배치해야 합니다.

## 핵심 피처 엔지니어링

- 날짜 파생: 출생일, 도축일, 판정일 기반 연/월/계절/기간 변수
- 체중·나이 파생: `weight_per_age`, age/weight bin, 상호작용 변수
- 범주형 빈도 인코딩: 농장, 지역, KPN, 부모/조부모 계통, 조합 key
- 지역/농장 밀도 파생: 농장 수, 면적, 밀도
- 기상 결합: 관측지점 및 날짜 기준 기온, 강수량, 습도, 풍속
- 혈통/KPN 결합: `CATTLE_NO`, `KPN_NO` 기준 유전능력 정보 연결
- history feature: 농장/지역/KPN/성별 조합별 과거 등급 평균, A/B/C 비율, 상위 등급 비율
- 확률 후처리: C계열 과소예측 완화를 위한 suffix probability boost

## 모델링 방식

최종 모델은 CatBoost와 XGBoost를 각각 학습한 뒤 예측 확률을 결합했습니다.

```text
final_proba = 0.6 * CatBoost_proba + 0.4 * XGBoost_proba
```

이후 temperature scaling과 C계열 확률 보정을 적용했습니다.

## 주요 실험 결과

| 실험 | Public Macro-F1 |
|---|---:|
| Cat+XGB base fullhist | 0.210 |
| seed ensemble | 0.214 |
| best80 + colab1m adjusted20 | 0.218 |
| colab1m cboost115 adjusted 단독 | 0.219 |
| colab1m cboost118 adjusted 단독 | 0.219 |

## 배운 점

한우 등급 예측은 단순 분류 모델 성능 문제가 아니라, 도메인 결정 구조를 이해하고 이를 피처와 모델 구조에 반영해야 하는 문제였습니다. 특히 train에만 존재하는 도축 후 판정 변수의 누수 가능성을 제거하고, test에서 실제 활용 가능한 정보만으로 예측 구조를 재설계하는 과정이 중요했습니다.

또한 16개 클래스를 직접 예측하는 방식만으로는 A/B/C 세부 등급, 특히 C계열을 안정적으로 맞히기 어려웠습니다. 향후에는 품질축과 수율축을 분리한 multi-stage 모델, C-risk 모델, 체중-나이-성별 상대지표, 누적 기상 스트레스 파생변수를 추가로 설계할 필요가 있습니다.

## 주의

본 저장소에는 공모전 원천 데이터와 대용량 제출 파일을 포함하지 않습니다. 데이터 사용권과 파일 용량 문제를 고려하여 코드와 분석 기록 중심으로 정리했습니다.
