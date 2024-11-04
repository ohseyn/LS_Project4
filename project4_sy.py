import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

df_dc = pd.read_csv("data_week4.csv", encoding='cp949')
df = df_dc.copy()
df.columns = ['Unnamed: 0', '작업라인', '제품명', '금형명', '수집날짜', '수집시각', '일자별 제품 생산 번호',
              '가동여부', '비상정지', '용탕온도', '설비 작동 사이클 시간', '제품 생산 사이클 시간',
              '저속구간속도', '고속구간속도', '용탕량', '주조압력', '비스킷 두께', '상금형온도1',
              '상금형온도2', '상금형온도3', '하금형온도1', '하금형온도2', '하금형온도3', '슬리브온도',
              '형체력', '냉각수 온도', '전자교환 가동시간', '등록일시', '불량판정', '사탕신호', '금형코드',
              '가열로']
df = df.drop(columns=['Unnamed: 0'])
#=================================================
df.describe()
df.head()
print(df.shape)
print(df.columns)
print(df.isnull().sum())
df.info()
#=================================================
# 결측값 시각화
import missingno as msno
msno.matrix(df)

# 숫자형 데이터만 선택하여 정보 확인
numeric_columns = df.select_dtypes(include=['float64', 'int64'])
object_columns = df.select_dtypes(include="object")

# 숫자형 열의 정보를 출력
print(numeric_columns.info())
print(object_columns.info())
#==================================================
# 가동여부의 고유값 개수 확인
operation_status_counts = df["가동여부"].value_counts(dropna=False)

# 고유값과 그 개수 출력
print(operation_status_counts)

# '가동여부'가 '정지'이거나 NaN인 행 필터링
filtered_df = df[(df['가동여부'] == '정지') | (df['가동여부'].isna())]

# 해당 행들의 날짜 확인 (예: '수집날짜'가 날짜 열이라고 가정)
print(filtered_df[['수집날짜', '수집시각', '가동여부']])

# 가동여부가 '정지' 또는 NaN인 부분을 필터링
df['수집날짜'] = pd.to_datetime(df['수집날짜'])  # 날짜 형식 변환
df = df.sort_values(by='수집날짜')  # 날짜 기준 정렬

# 전체 시계열 데이터를 확인 (예: '수집날짜' 기준)
plt.figure(figsize=(12, 6))

# 전체 가동여부 시계열 데이터 (가동: 파란색)
plt.plot(df['수집날짜'], df['가동여부'] == '가동', label='가동', color='blue', alpha=0.5)

# 정지인 부분 강조 (빨간색으로 강조)
plt.scatter(df.loc[df['가동여부'] == '정지', '수집날짜'],
            [True] * len(df.loc[df['가동여부'] == '정지']), color='red', label='정지', zorder=2)

# NaN인 부분 강조 (노란색으로 강조)
plt.scatter(df.loc[df['가동여부'].isna(), '수집날짜'],
            [True] * len(df.loc[df['가동여부'].isna()]), color='yellow', label='NaN', zorder=2)

# 그래프 제목 및 범례
plt.title('가동 여부 시계열 데이터 (정지 및 NaN 강조)', fontsize=16)
plt.xlabel('수집날짜')
plt.ylabel('가동 여부')
plt.legend()
plt.show()
#===============================================
# 범주형, 문자형 drop
df_corr = df.drop(columns=['작업라인', '제품명', '금형명', '수집날짜', '수집시각', '가동여부', '비상정지',
                        '등록일시', '사탕신호', '가열로', '설비 작동 사이클 시간', '제품 생산 사이클 시간', 
                        '일자별 제품 생산 번호', '전자교환 가동시간', '불량판정', '금형코드'])

# 스피어만 상관계수 계산 (결측치 처리하지 않고 그대로 유지)
spearman_corr = df_corr.corr(method='spearman')

rc('font', family='Malgun Gothic')

# 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('스피어만 상관관계 히트맵 (결측치 포함)')
plt.show()

# 날짜 변환
df['수집날짜'] = pd.to_datetime(df['수집날짜'])

# 숫자형 데이터만 선택하고, 날짜를 기준으로 일별 평균 계산
df_numeric = df.select_dtypes(include=['float64', 'int64'])  # 숫자형 데이터만 선택
df_numeric.index = df['수집날짜']  # 날짜를 인덱스로 설정

# 일별 평균으로 재샘플링 후 시각화
df_numeric.resample('D').mean().plot(figsize=(12, 6))
plt.title('일별 평균 데이터')
plt.show()
#====================================================
import holidays
kr_holidays = holidays.KR()

df["날짜_시간"] = pd.to_datetime(df["등록일시"])
df['월'] = df.날짜_시간.dt.month                    # 월(숫자)
df['일'] = df.날짜_시간.dt.day                        # 일(숫자)
df['시'] = df.날짜_시간.dt.hour                      # 시(숫자)
df['요일'] = df.날짜_시간.dt.weekday                # 요일(숫자)
df['주말'] = df['날짜_시간'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일

df.drop(["작업라인", "제품명", "금형명", "수집날짜", "수집시각", "비상정지", "날짜_시간"], axis = 1)

plt.figure(figsize=(14, 6))

# 날짜별 불량 판정 평균 계산
daily_trend = df.groupby(df['날짜_시간'].dt.date)['불량판정'].mean().reset_index()

# 그래프 그리기
sns.lineplot(data=daily_trend, x='날짜_시간', y='불량판정')
plt.title('1월 2일부터 3월 31일까지의 날짜별 불량 판정 추세')
plt.xlabel('날짜')
plt.ylabel('불량 판정 평균')
plt.xticks(rotation=45)  # x축 레이블 회전
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.show()
#================================================
# '등록일시'를 datetime 형식으로 변환
df['등록일시'] = pd.to_datetime(df['등록일시'])

# 날짜와 시간 정보 추출
df['수집_날짜'] = df['등록일시'].dt.date  # 날짜만 추출
df['수집_시간'] = df['등록일시'].dt.hour  # 시간만 추출

# '불량판정' 열의 결측치를 0으로 대체
df['불량판정'] = df['불량판정'].fillna(0).astype(int)

# 시간대별 불량률 계산
hourly_failure_rate = df.groupby('수집_시간')['불량판정'].mean()

# 요일별 불량률 계산
df['수집_요일'] = df['등록일시'].dt.day_name()
daily_failure_rate = df.groupby('수집_요일')['불량판정'].mean()

# 시각화: 시간대별 불량률
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
hourly_failure_rate.plot(kind='bar', color='skyblue')
plt.title('시간대별 불량률')
plt.xlabel('시간대')
plt.ylabel('평균 불량률')

# 시각화: 요일별 불량률
plt.subplot(1, 2, 2)
daily_failure_rate.plot(kind='bar', color='lightgreen')
plt.title('요일별 불량률')
plt.xlabel('요일')
plt.ylabel('평균 불량률')

plt.tight_layout()
plt.show()
#============================================
# 숫자형 데이터만 선택 (범주형/문자열 데이터를 제외)
numeric_df = df.select_dtypes(include=['float64', 'int64'])

numeric_df = numeric_df.drop()

# 불량판정(타겟)과 공정 변수들 간의 상관관계 계산
correlation_with_fail = numeric_df.corr()['불량판정'].sort_values(ascending=False)

# 상관관계 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_fail.index, y=correlation_with_fail.values)
plt.xticks(rotation=90)
plt.title('불량판정과 공정 변수들의 상관관계')
plt.show()
#===========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(x='주조압력', y='비스킷 두께', hue='불량판정', data=df)
plt.title('주조압력과 비스킷 두께 조합에 따른 불량 여부')
plt.show()
#============================================
# 1. 불량률이 높은 날짜를 선택 (여기서는 예시로 특정 날짜를 지정)
# 실제로 불량률이 높은 날짜를 찾으려면 날짜별 불량률을 계산해야 합니다.
high_failure_date = '2019-03-31'  # 예시 날짜 (불량률이 높은 날짜로 교체하세요)

# 2. 해당 날짜의 데이터를 필터링
df['수집날짜'] = pd.to_datetime(df['수집날짜'])  # 날짜 형식 변환
selected_date_df = df[df['수집날짜'] == high_failure_date]

# 3. 불량일 때와 불량이 아닐 때의 데이터를 나누기
fail_data = selected_date_df[selected_date_df['불량판정'] == 1]  # 불량 데이터
pass_data = selected_date_df[selected_date_df['불량판정'] == 0]  # 양품 데이터

# 4. 설명변수들의 평균 계산 (숫자형 변수만 대상으로)
fail_mean = fail_data.select_dtypes(include=['float64', 'int64']).mean()
pass_mean = pass_data.select_dtypes(include=['float64', 'int64']).mean()

# 5. 불량일 때와 불량이 아닐 때의 평균 비교
comparison_df = pd.DataFrame({'불량 평균': fail_mean, '양품 평균': pass_mean})

# 6. 결과 출력
print(comparison_df)