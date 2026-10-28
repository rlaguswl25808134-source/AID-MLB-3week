# 혼공머신 3장 — 회귀 알고리즘과 모델 규제

『혼자 공부하는 머신러닝 + 딥러닝』 3장을 공부하고 정리한 학습 노트 사이트입니다.
교재의 농어(perch) 데이터 대신 **핀치부리새 데이터(부리 길이 → 부리 높이)** 로 3-1, 3-2 실습을 다시 했고,
3-3은 실습 없이 개념만 정리했습니다.

- 정적 사이트: 빌드 도구 없이 `index.html` 하나로 동작
- 차트: 외부 라이브러리 없이 인라인 SVG로 직접 그림
- 실습 수치: 노트북과 같은 `train_test_split(random_state=42)` 분할을 브라우저에서 재현

## 구성

| 섹션 | 내용 |
| --- | --- |
| Overview | 3장 전체 로드맵, 무엇을 배웠는지 |
| Data | 핀치부리새 49마리, 훈련 36 / 테스트 13 분할 |
| KNN | k를 바꿔 가며 예측 곡선과 R², MAE 확인 |
| Linear | 선형 회귀 계수·절편, 30.0 예측값 |
| Features | 다항 회귀(2차)와 과대/과소적합 |
| Compare | 세 모델 비교, 교재 수치와의 차이 |
| Glossary | 헷갈리는 용어 정리 |
| Code | 노트북 핵심 코드 |
| Quiz | 6문항 자가 점검 |
| Notes | 손글씨 필기 이미지 원본 |

## 파일 구조

```
site/
├─ index.html          # 모든 콘텐츠
├─ assets/
│  ├─ style.css        # 디자인 시스템, 반응형
│  ├─ app.js           # SVG 차트, KNN/선형/다항 계산, 탭·퀴즈·라이트박스
│  ├─ data.js          # 핀치부리새 49개 데이터 (훈련/테스트)
│  └─ notes-3-1.jpg    # 3-1 필기 스캔본
└─ README.md
```

## 로컬에서 보기

`file://` 로 열어도 대부분 동작하지만, 폰트·이미지 경로 문제를 피하려면 간단한 서버를 권합니다.

```powershell
cd site
python -m http.server 8000
# http://localhost:8000/
```

## GitHub Pages 배포

1. 저장소에 `site/` 안의 파일을 올립니다. (저장소 루트에 `index.html`이 오도록)
2. **Settings → Pages → Build and deployment → Source: Deploy from a branch**
3. Branch `main`, 폴더 `/ (root)` 선택 후 저장
4. 잠시 뒤 `https://<사용자명>.github.io/<저장소명>/` 에서 확인

## 참고한 자료

- 교재: 『혼자 공부하는 머신러닝 + 딥러닝』 3장 (PDF 원본은 저작권 문제로 저장소에 포함하지 않았습니다)
- 실습 노트북
  - [2-2](https://github.com/rlaguswl25808134-source/AID-MLB/blob/MLB-2-2/2-2.ipynb)
  - [3-1](https://github.com/rlaguswl25808134-source/AID-MLB/blob/MLB-3-1/3-1.ipynb)
  - [3-2](https://github.com/rlaguswl25808134-source/AID-MLB/blob/MLB-3-2/3-2.ipynb)
- 3-1 손글씨 필기 이미지

## 수치 메모

- KNN k=3은 거리가 같은 이웃이 생기는 지점이 있어, sklearn 내부 선택 순서에 따라 R² 소수점 넷째 자리가 달라집니다(0.9874 ↔ 0.9875).
  표와 코드 주석에는 노트북 출력값을, 인터랙티브 계산기는 브라우저 계산값을 표시합니다.
- 핵심 수치(k=5): 훈련 R² 0.9879, 테스트 R² 0.9817, 테스트 MAE 0.1615
- 선형 회귀: coef −0.6069, intercept 11.7460, 길이 30.0 → −6.46
- 다항 회귀: coef [−0.0743, 1.1265], intercept 2.0916, 길이 30.0 → −30.94
