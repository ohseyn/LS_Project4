import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_csv('data_week4.csv', encoding='cp949')

# 데이터 확인
df.head()
df.columns
df.info()
df.isnull().sum()
df.nunique()  

# 데이터명 변경
df.columns = ['Unnamed:0', '작업라인', '제품명', '금형명', '수집날짜', '수집시각', '일자별제품생산번호',
              '가동여부', '비상정지', '용탕온도', '설비작동사이클시간', '제품생산사이클시간',
              '저속구간속도', '고속구간속도', '용탕량', '주조압력', '비스킷두께', '상금형온도1',
              '상금형온도2', '상금형온도3', '하금형온도1', '하금형온도2', '하금형온도3', '슬리브온도',
              '형체력', '냉각수온도', '전자교반가동시간', '등록일시', '불량판정', '사탕신호', '금형코드',
              '가열로']

# 결측치 30% 이상인 행 제거
df = df[df.isnull().mean(axis=1) * 100 < 30]

# df[df.isnull().mean(axis=1) * 100 >= 30]

# '사탕신호'가 'D'인 행 Drop
df = df[df['사탕신호'] != 'D']

# 열 Drop (등록일시 포함)
df = df.drop(columns=['Unnamed:0', '일자별제품생산번호', '작업라인', '사탕신호', '제품명', '금형명', '비상정지', '수집날짜', '수집시각', '등록일시'])

# '불량판정', '금형코드' 열을 범주형으로 변환
df['불량판정'] = df['불량판정'].astype('category')
df['금형코드'] = df['금형코드'].astype('category')

# '가동여부' 변환: 가동이면 0, 아니면 1
df['가동여부'] = df['가동여부'].apply(lambda x: 0 if x == '가동' else 1)

# 가열로, 용탕온도, 용탕량, 하금형온도, 상금형온도3에서 결측치 발견
df.isna().sum()

# '가열로' 열의 'NaN' 값을 'F'(측정X)로 변경
df['가열로'] = df['가열로'].fillna('F')

# 용탕온도, 용탕량, 하금형온도3에 대해 선형 보간을 적용
df['용탕온도'] = df['용탕온도'].interpolate(method='linear', limit_direction='both')
df['용탕량'] = df['용탕량'].interpolate(method='linear', limit_direction='both')
df['하금형온도3'] = df['하금형온도3'].interpolate(method='linear', limit_direction='both')
df['상금형온도3'] = df['상금형온도3'].interpolate(method='linear', limit_direction='both')

df[["상금형온도3", "하금형온도3", "용탕온도", "용탕량"]].isnull().sum()

df.isna().sum()

# 이상치 제거
df = df[df['설비작동사이클시간'] <= 400] # 1
df = df[df['제품생산사이클시간'] <= 450] # 2
df = df[df['저속구간속도'] <= 60000] # 1
df = df[df['상금형온도1'] <= 1400] # 1
df = df[df['상금형온도2'] <= 4000] # 1
df = df[df['하금형온도3'] <= 60000] # 1
df = df[df['형체력'] <= 60000] # 3
df = df[df['냉각수온도'] <= 1400] # 9

#================================================================================================
# 필요한 컬럼만 추출
columns_needed = ['불량판정', '주조압력', '상금형온도1', '상금형온도2', '하금형온도1', '하금형온도2']
df_selected = df[columns_needed]

# 불량 여부에 따른 평균과 중앙값 각각 계산
mean_values = df_selected.groupby('불량판정').mean()
median_values = df_selected.groupby('불량판정').median()

# 평균값과 중앙값을 melt 형식으로 변환
mean_melted = mean_values.melt(var_name='변수', value_name='평균값', ignore_index=False)
median_melted = median_values.melt(var_name='변수', value_name='중앙값', ignore_index=False)

# 불량 여부를 다시 인덱스에서 컬럼으로 변환
mean_melted.reset_index(inplace=True)
median_melted.reset_index(inplace=True)

# 평균값과 중앙값을 하나의 데이터프레임으로 결합
combined_df = pd.merge(mean_melted, median_melted, on=['불량판정', '변수'])

# 불량 여부에 따른 값을 보기 좋게 표시
combined_pivot = combined_df.pivot(index='변수', columns='불량판정', values=['평균값', '중앙값'])
# 다중 인덱스 구조에서 열 이름을 새로 설정하는 방법
combined_pivot.columns = ['평균값 (정상)', '평균값 (불량)', '중앙값 (정상)', '중앙값 (불량)']

# 결과 출력
combined_pivot = combined_pivot.reset_index(drop=True)
combined_pivot.insert(0, '변수 이름', ["상금형온도1", "상금형온도2", "주조압력", "하금형온도1", "하금형온도2"])

#=======================================================
numeric_df = df[['용탕온도', '설비작동사이클시간', '제품생산사이클시간',
              '저속구간속도', '고속구간속도', '용탕량', '주조압력', '비스킷두께', '상금형온도1',
              '상금형온도2', '상금형온도3', '하금형온도1', '하금형온도2', '하금형온도3', '슬리브온도',
              '형체력', '냉각수온도', '전자교반가동시간']]
category_df = df[['가동여부', '금형코드', '가열로']]

# 수치형 변수들에 대한 박스플롯을 서브플롯으로 그리는 함수
def boxplot_subplots(df):
    # 수치형 변수 선택 (int64와 float64 타입만 선택)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns  
    
    # 서브플롯 행 계산 (한 행에 2개의 서브플롯 배치)
    total_plots = len(numeric_cols)  # 각 변수당 1개의 플롯
    max_cols = 2  # 한 행에 2개의 서브플롯
    max_rows = int(np.ceil(total_plots / max_cols))  # 행 개수 계산
    
    plt.figure(figsize=(15, 6 * max_rows))  # 전체 플롯 크기 설정
    
    plot_index = 1
    for col in numeric_cols:
        # 각 변수에 대해 하나의 박스플롯 그리기
        plt.subplot(max_rows, max_cols, plot_index)
        sns.boxplot(data=df, y=col, palette='Set2')
        plt.title(f'{col}의 박스플롯', fontsize=12)
        plt.ylabel(col, fontsize=10)
        plt.xticks(rotation=90)
        plot_index += 1
    
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.tight_layout()  # 서브플롯 간의 간격 조정
    plt.show()

# 박스플롯 그리기
boxplot_subplots(df)

#========================================================
sns.countplot(data = df, x = '가열로', hue = '불량판정', palette='Blues')
# 그래프 제목 및 라벨 설정
plt.title('가열로별 불량판정 Barplot')
plt.xlabel('가열로')
plt.ylabel('Count')
plt.legend(title='불량판정')
plt.show()

# 교차표를 사용하여 가열로별 불량판정의 개수를 계산
count_df = pd.crosstab(df['가열로'], df['불량판정'])

# 퍼센티지를 계산하기 위해 각 행의 합을 기준으로 퍼센티지로 변환
percentage_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

# 결과를 데이터프레임으로 병합하여 확인
combined_df = pd.concat([count_df, percentage_df], axis=1, keys=['Count', 'Percentage'])