import time
import win32com.client
import pandas as pd
import numpy as np


def get_all_index_names():
    objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")
    all_index_names = []

    for category in range(6):  # 보통 0~5번 인덱스 그룹 존재
        try:
            index_names = objIndex.GetChartIndexCodeListByIndex(category)
            all_index_names.extend(index_names)
        except Exception as e:
            print(f"카테고리 {category} 처리 중 오류 발생:", e)

    return all_index_names




def get_macd(stock_code='A000660', cnt=100):
    # 1. 주가 데이터 요청 (CpSysDib.StockChart)
    objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
    objStockChart.SetInputValue(0, stock_code)       # 종목 코드
    objStockChart.SetInputValue(1, ord('2'))         # 개수로 조회
    objStockChart.SetInputValue(4, cnt)              # 최근 cnt개
    objStockChart.SetInputValue(5, [0, 2, 3, 4, 5, 8]) # 날짜, 시가, 고가, 저가, 종가, 거래량
    objStockChart.SetInputValue(6, ord('D'))         # 일별
    objStockChart.SetInputValue(9, ord('1'))         # 수정주가
    objStockChart.BlockRequest()

    len_data = objStockChart.GetHeaderValue(3)
    close_list = []

    for i in range(len_data):
        close_price = objStockChart.GetDataValue(4, len_data - 1 - i)
        close_list.append(close_price)

    # 2. CpIndexes.CpSeries에 종가 데이터 추가
    objSeries = win32com.client.Dispatch("CpIndexes.CpSeries")
    for close in close_list:
        objSeries.Add(close, 0, 0, 0, 0)  # close, open, high, low, volume

    # 3. CpIndexes.CpIndex로 MACD 계산
    objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")
    objIndex.series = objSeries
    objIndex.put_IndexKind("MACD")         # MACD 지표 선택
    objIndex.put_IndexDefault("MACD")      # 기본 조건값 적용 (12, 26, 9)
    objIndex.Calculate()

    macd = [objIndex.GetResult(0, i) for i in range(objIndex.GetCount(0))]
    signal = [objIndex.GetResult(1, i) for i in range(objIndex.GetCount(1))]
    osc = [objIndex.GetResult(2, i) for i in range(objIndex.GetCount(2))]

    return macd, signal, osc

#=============================================위는 지표코드

def get_pattern(df):
    df['우산형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['망치형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['교수형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['장악형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['흑운형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['관통형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['샛별형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['저녁별형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['십자샛별형/십자저녁별형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['유성형/역망치형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['유성형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['역망치형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['잉태형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['십자잉태형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['집게형 천장/바닥'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['샅바형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['까마귀형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['흑삼병'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['상승적삼병'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['삼산형/삼천형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['반격형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['고기만두형/프라이팬형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['탑형 천장/바닥'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['창형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['타스키형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['고가/저가 갭핑 플레이'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['나란히형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['상승삼법형/하락삼법형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['갈림길형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['북향 도지형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['키다리형, 비석형, 잠자리형 도지'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    df['삼별형'] = np.random.choice([1, 0], p=[0.0005, 0.9995])
    return df

def get_sub(df):
    df.sort_values(by='날짜', ascending=True).reset_index(drop=True)
    #이동평균선
    df['종5일이동평균'] = df['종가'].rolling(window=5).mean()
    df['종10일이동평균'] = df['종가'].rolling(window=10).mean()
    df['종20일이동평균'] = df['종가'].rolling(window=20).mean()
    df['종60일이동평균'] = df['종가'].rolling(window=60).mean()
    df['거5일이동평균'] = df['거래량'].rolling(window=5).mean()
    df['거10일이동평균'] = df['거래량'].rolling(window=10).mean()
    df['거20일이동평균'] = df['거래량'].rolling(window=20).mean()
    df['거60일이동평균'] = df['거래량'].rolling(window=60).mean()
    #MACD
    ema12 = df['종가'].ewm(span=12, adjust=False).mean()
    ema26 = df['종가'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal
    #rsi
    diff = df['종가'].diff()
    up_diff = diff.copy()
    down_diff = diff.copy()
    up_diff[up_diff < 0] = 0
    down_diff[down_diff > 0] = 0
    up_diff = up_diff.rolling(window=14).mean()
    down_diff = down_diff.rolling(window=14).mean()# 평균 상승폭 / 평균 하락폭(절대값)
    df['RSI'] = (up_diff / -down_diff) / ((up_diff / -down_diff) + 1) * 100
    #스토캐스틱
    highest5days = df['고가'].rolling(window=5).max()
    lowest5days = df['저가'].rolling(window=5).min()
    df['스토캐스틱'] = (df['종가'] - lowest5days) / (highest5days - lowest5days) * 100
    #OBV
    df['OBV'] = 0
    for i in range(1, len(df)):
        if df['종가'].iloc[i] >= df['종가'].iloc[i - 1]:
            df.loc[i, 'OBV'] = df.loc[i - 1, 'OBV'] + df.loc[i, '거래량']
        else:
            df.loc[i, 'OBV'] = df.loc[i - 1, 'OBV'] - df.loc[i, '거래량']
    #몸통대비
    df['Body'] = (df['종가'] - df['시가']).abs()
    df['Range'] = df['고가'] - df['저가']
    df['Body_vs_Range'] = df['Body'] / df['Range'].replace(0, pd.NA)

    df['UpperShadow'] = df['고가'] - df[['시가', '종가']].max(axis=1)
    df['LowerShadow'] = df[['시가', '종가']].min(axis=1) - df['저가']

    df['UpperRatio'] = df['UpperShadow'] / df['Body'].replace(0, pd.NA)
    df['LowerRatio'] = df['LowerShadow'] / df['Body'].replace(0, pd.NA)
    df.drop(columns=["Body"], inplace=True)
    df.drop(columns=["Range"], inplace=True)
    # VR(Volume Ratio) 계산 (14일 기준)
    df['price_change'] = df['종가'].diff()
    df['up_vol'] = df['거래량'].where(df['price_change'] > 0, 0)
    df['down_vol'] = df['거래량'].where(df['price_change'] < 0, 0)
    df['VR'] = (df['up_vol'].rolling(14).sum() / df['down_vol'].rolling(14).sum()) * 100
    df.drop(columns=["price_change"], inplace=True)
    df.drop(columns=["up_vol"], inplace=True)
    df.drop(columns=["down_vol"], inplace=True)

    df.sort_values(by='날짜', ascending=False).reset_index(drop=True)
    return df

#==================위는 패턴, 지표코드

def fetch_historical_data_with_delay(stock_code, start_date, end_date):
    objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
    total_data = []

    objStockChart.SetInputValue(0, stock_code)  # 종목코드
    objStockChart.SetInputValue(1, ord('1'))  # 개수로 요청
    objStockChart.SetInputValue(2, end_date)  # 종료일 (YYYY1231)
    objStockChart.SetInputValue(3, start_date)  # 시작일 (YYYY0101)
    objStockChart.SetInputValue(5, (0, 2, 3, 4, 5, 8))  # 필드: 날짜, 시가, 고가, 저가, 종가, 거래량
    objStockChart.SetInputValue(6, ord('D'))  # 일간 데이터
    objStockChart.SetInputValue(9, ord('1'))  # 수정주가 사용
        # 데이터 요청
    objStockChart.BlockRequest()

    num_data = objStockChart.GetHeaderValue(3)  # 수신된 데이터 개수

        # 데이터 수집
    for i in range(num_data):
        date = objStockChart.GetDataValue(0, i)  # 날짜
        open_price = objStockChart.GetDataValue(1, i)  # 시가
        high_price = objStockChart.GetDataValue(2, i)  # 고가
        low_price = objStockChart.GetDataValue(3, i)  # 저가
        close_price = objStockChart.GetDataValue(4, i)  # 종가
        volume = objStockChart.GetDataValue(5, i)  # 거래량
        total_data.append((date, open_price, high_price, low_price, close_price, volume))
        #

    return total_data

def get_kosdaq_stock_codes():
    objCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
    kosdaq_codes = objCpCodeMgr.GetStockListByMarket(2)  # 2는 코스닥 시장
    return kosdaq_codes
'''
if df.empty or df["종가"].max() == 0:
    continue  # 종가 0이면 오류 방지

outstanding_shares = capVal / df["종가"].iloc[0]  # 기준 종가로 나눔

# 조건에 맞는 행만 남기기: 종가 * 발행주식수 >= 5000억
df["시가총액"] = df["종가"] * outstanding_shares
df_filtered = df[df["시가총액"] >= 5000 * 1e8].copy()  # 5000억 이상만 유지

# 필요 없어진 시가총액 컬럼 삭제
df_filtered.drop(columns=["시가총액"], inplace=True)
'''
def get_market_cap(code):
    objMarketEye = win32com.client.Dispatch("CpSysDib.MarketEye")
    objMarketEye.SetInputValue(0, (67, 4))  # 67: 시가총액(억원), 4: 현재가
    objMarketEye.SetInputValue(1, code)
    objMarketEye.BlockRequest()
    market_cap = objMarketEye.GetDataValue(0, 0)
    return market_cap




def list_all_index_and_conditions():
    objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")

    print("📈 지원 지표 목록과 기본 조건값")
    print("=" * 60)

    for category in range(6):  # 지표 카테고리 인덱스 0 ~ 5
        try:
            index_list = objIndex.GetChartIndexCodeListByIndex(category)
            for index_name in index_list:
                try:
                    objIndex.put_IndexKind(index_name)
                    objIndex.put_IndexDefault(index_name)
                    term1 = objIndex.get_Term1()
                    term2 = objIndex.get_Term2()
                    term3 = objIndex.get_Term3()
                    term4 = objIndex.get_Term4()
                    signal = objIndex.get_Signal()
                    print(f"{index_name:<25} | 조건1: {term1:<3} 조건2: {term2:<3} 조건3: {term3:<3} 조건4: {term4:<3} Signal: {signal}")
                except Exception as e:
                    print(f"⚠️ '{index_name}' 지표 조건 불러오기 실패: {e}")
        except Exception as e:
            print(f"⚠️ 카테고리 {category} 지표 리스트 불러오기 실패: {e}")

'''
if __name__ == "__main__":
    objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
    kosdaq_codes = get_kosdaq_stock_codes()
    finalcode=['A000250', 'A000440', 'A003100', 'A003800', 'A005290', 'A005670', 'A007330', 'A007390', 'A008830', 'A009300', 'A009520', 'A009780', 'A011560', 'A013030', 'A014620', 'A017890', 'A018120', 'A018290', 'A020400', 'A023160', 'A023900', 'A023910', 'A024060', 'A025770', 'A025870', 'A025950', 'A025980', 'A028300', 'A029960', 'A030520', 'A031980', 'A032190', 'A032300', 'A032685', 'A032960', 'A033100', 'A033160', 'A033500', 'A034950', 'A035760', 'A035900', 'A036190', 'A036480', 'A036800', 'A036810', 'A036830', 'A036890', 'A036930', 'A037460', 'A038290', 'A039030', 'A039200', 'A039440', 'A039610', 'A039840', 'A041510', 'A041830', 'A042000', 'A042370', 'A042420', 'A042510', 'A043150', 'A045100', 'A045660', 'A047920', 'A048410', 'A048910', 'A049070', 'A049430', 'A049720', 'A049950', 'A051160', 'A051500', 'A052020', 'A052400', 'A053030', 'A053080', 'A053160', 'A053580', 'A053610', 'A053800', 'A054450', 'A054930', 'A056190', 'A058470', 'A058610', 'A058970', 'A059090', 'A060370', 'A064290', 'A064550', 'A064760', 'A064820', 'A064850', 'A065350', 'A065660', 'A065680', 'A065710', 'A066620', 'A067080', 'A067160', 'A067280', 'A067370', 'A068760', 'A068930', 'A069510', 'A071280', 'A072020', 'A072870', 'A073490', 'A074600', 'A078070', 'A078340', 'A078600', 'A079940', 'A079960', 'A080010', 'A080220']
    print(len(finalcode))
    i=0
'''
'''

    for code in kosdaq_codes[:630]:
        capVal=get_market_cap(code)#시가총액 받아와서 필터링
        i+=1
        print(i)
        if capVal<5000:
            print("skip: "+code)
            continue
        #index_list = get_all_index_names()
        #for name in index_list:
        #    print(name)
        #print(len(index_list))
        #list_all_index_and_conditions()
        #data=get_stock_data_with_macd(code, 20150414, 20250414)
        data = fetch_historical_data_with_delay(code, 20150414, 20250414)
        columns = ["날짜", "시가", "고가", "저가", "종가", "거래량"]
        df = pd.DataFrame(data, columns=columns)
        df["종가"] = pd.to_numeric(df["종가"], errors='coerce')
        if df.empty or df["종가"].dropna().empty:
            print("❌ Skip (종가 없음 또는 모두 NaN):", code)
            continue

        current_price = df["종가"].iloc[-1]  # 가장 최근 종가
        if current_price == 0:
            print("❌ Skip (종가 0):", code)
            continue

        # 주식 수 = 시가총액 / 현재 종가
        share_count = capVal * 1e8 / current_price  # 억 원 → 원으로 환산

        # 기준 주가 = 5000억 / 주식 수
        target_price = 5000 * 1e8 / share_count

        if (df["종가"] < target_price).any():
            print("❌ Skip (종가 중 기준 주가보다 낮은 시점 존재):", code)
            continue

        print("✅ PASS:", code)
        finalcode.append(code)
        df.to_csv("D:\\졸업프로젝트\\데이터\\"+code+".csv", index=False, encoding='utf-8-sig')

        time.sleep(3.6)
    print(finalcode)
    finalcode.to_csv("D:\\졸업프로젝트\\데이터\\filteredCode.csv", index=False, encoding='utf-8-sig')

    
        df = get_sub(df)
        print("지표계산열 추가")
        print(df.head())
        #지표계산열 추가

        print("패턴열 추가")
        df = get_pattern(df)
        print(df.head())
        #패턴열 추가
        
        #df.to_csv("D:\\졸업프로젝트\\데이터\\"+code+".csv", index=False, encoding='utf-8-sig')
        time.sleep(0.3)  # 요청 간 딜레이'''
#=============================아래로 지표 150개 코드

def get_all_index_names():
    objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")
    all_index_names = []

    for category in range(6):
        try:
            index_list = objIndex.GetChartIndexCodeListByIndex(category)
            all_index_names.extend(index_list)
        except:
            continue
    return all_index_names

def get_ohlcv_series(stock_code, cnt):
    # OHLCV 데이터 수집
    chart = win32com.client.Dispatch("CpSysDib.StockChart")
    chart.SetInputValue(0, stock_code)
    chart.SetInputValue(1, ord('2'))  # 개수 기준
    chart.SetInputValue(4, cnt)
    chart.SetInputValue(5, [0, 2, 3, 4, 5, 8])  # 날짜, 시가, 고가, 저가, 종가, 거래량
    chart.SetInputValue(6, ord('D'))
    chart.SetInputValue(9, ord('1'))
    chart.BlockRequest()

    count = chart.GetHeaderValue(3)
    ohlcv = []
    dates = []

    for i in range(count):
        date = chart.GetDataValue(0, count - 1 - i)
        open_ = chart.GetDataValue(1, count - 1 - i)
        high = chart.GetDataValue(2, count - 1 - i)
        low = chart.GetDataValue(3, count - 1 - i)
        close = chart.GetDataValue(4, count - 1 - i)
        volume = chart.GetDataValue(5, count - 1 - i)

        objSeries.Add(close, open_, high, low, volume)
        ohlcv.append([date, open_, high, low, close, volume])
        dates.append(date)

    df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format='%Y%m%d')
    df.set_index("Date", inplace=True)

    return objSeries, df

def calculate_all_indexes(series, index_names):
    index_data = {}
    objIndex.series = series

    for idx, name in enumerate(index_names):
        if idx in [103,160]:
            continue
        try:
            #print(f"[{idx}/{len(index_names)}] ⏳ {name} 계산 중...")
            objIndex.put_IndexKind(name)
            objIndex.put_IndexDefault(name)
            objIndex.series = objSeries
            objIndex.Calculate()

            count = objIndex.ItemCount
            for i in range(count):
                result = [objIndex.GetResult(i, j) for j in range(objIndex.GetCount(i))]
                #print(f"✅ {name} 계산 완료: {len(result)}개")
                index_data[name] = result
        except Exception as e:
            #print(f"❌ {name} 실패: {e}")
            continue

    return index_data

def is_index_calculable(index_name, objSeries):
    try:
        objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")
        objIndex.series = objSeries
        objIndex.put_IndexKind(index_name)
        objIndex.put_IndexDefault(index_name)
        objIndex.Calculate()
        return True
    except:
        return False

# 실행 예시
if __name__ == "__main__":
    codeList=['A000250', 'A000440', 'A003100', 'A003800', 'A005290', 'A005670', 'A007330', 'A007390', 'A008830', 'A009300', 'A009520', 'A009780', 'A011560', 'A013030', 'A014620', 'A017890', 'A018120', 'A018290', 'A020400', 'A023160', 'A023900', 'A023910', 'A024060', 'A025770', 'A025870', 'A025950', 'A025980', 'A028300', 'A029960', 'A030520', 'A031980', 'A032190', 'A032300', 'A032685', 'A032960', 'A033100', 'A033160', 'A033500', 'A034950', 'A035760', 'A035900', 'A036190', 'A036480', 'A036800', 'A036810', 'A036830', 'A036890', 'A036930', 'A037460', 'A038290', 'A039030', 'A039200', 'A039440', 'A039610', 'A039840', 'A041510', 'A041830', 'A042000', 'A042370', 'A042420', 'A042510', 'A043150', 'A045100', 'A045660', 'A047920', 'A048410', 'A048910', 'A049070', 'A049430', 'A049720', 'A049950', 'A051160', 'A051500', 'A052020', 'A052400', 'A053030', 'A053080', 'A053160', 'A053580', 'A053610', 'A053800', 'A054450', 'A054930', 'A056190', 'A058470', 'A058610', 'A058970', 'A059090', 'A060370', 'A064290', 'A064550', 'A064760', 'A064820', 'A064850', 'A065350', 'A065660', 'A065680', 'A065710', 'A066620', 'A067080', 'A067160', 'A067280', 'A067370', 'A068760', 'A068930', 'A069510', 'A071280', 'A072020', 'A072870', 'A073490', 'A074600', 'A078070', 'A078340', 'A078600', 'A079940', 'A079960', 'A080010', 'A080220','A082270', 'A082920', 'A083310', 'A083450', 'A083650', 'A083930', 'A084110', 'A084370', 'A085660', 'A086390', 'A086450', 'A086520', 'A086670', 'A086900', 'A087010', 'A088340', 'A089010', 'A089030', 'A089600', 'A089970', 'A089980', 'A090360', 'A092130', 'A092460', 'A092730', 'A093320', 'A093520', 'A094170', 'A094360', 'A094820', 'A094940', 'A095340', 'A095610', 'A095660', 'A096240', 'A096250', 'A096530', 'A098070', 'A098460', 'A099190', 'A099320', 'A099750', 'A100030', 'A100120', 'A101160', 'A101490', 'A101930', 'A101970', 'A102120', 'A102710', 'A102940', 'A104460', 'A104830', 'A106190', 'A107640', 'A108380', 'A108490', 'A108860', 'A109080', 'A109740', 'A109860', 'A110990', 'A112040', 'A112290', 'A114840', 'A115310', 'A115440', 'A115450', 'A115500', 'A119610', 'A119850', 'A120240', 'A121600', 'A122640', 'A122870', 'A123860', 'A126340', 'A126700', 'A131290', 'A131970', 'A136540', 'A137400', 'A138610', 'A140410', 'A140860', 'A141080', 'A143160', 'A143240', 'A144510', 'A145020', 'A148250', 'A160190', 'A160980', 'A161580', 'A163280', 'A166090', 'A168360', 'A171090', 'A173130', 'A174900', 'A178320', 'A179900', 'A182360', 'A183300', 'A186230', 'A187870', 'A189300', 'A190510', 'A191420', 'A194700', 'A195940', 'A196170', 'A199800', 'A200670', 'A204270', 'A206650', 'A211270', 'A213420', 'A214150', 'A214260', 'A214370', 'A214430', 'A214450', 'A215000', 'A215200', 'A215360', 'A218410', 'A219130', 'A220100', 'A221980', 'A222160', 'A222800', 'A224110', 'A225570', 'A226590', 'A226950', 'A228760', 'A230240', 'A232140', 'A232680', 'A236200', 'A237690', 'A239890', 'A240550', 'A240810', 'A241710', 'A243070', 'A247540', 'A251120', 'A251370', 'A251970', 'A253450', 'A253840', 'A254490', 'A256940', 'A257720', 'A260970', 'A263720', 'A263750', 'A263860', 'A264660', 'A265520', 'A267980', 'A270660', 'A272290', 'A274090', 'A277810', 'A278280', 'A281740', 'A282720', 'A282880', 'A285490', 'A289930', 'A290650', 'A295310', 'A298380', 'A299030', 'A303810', 'A304100', 'A304360', 'A308430', 'A310210', 'A314930', 'A317330', 'A319660', 'A323280', 'A323350', 'A323990', 'A328130', 'A335890', 'A336570', 'A336680', 'A340570', 'A347850', 'A348210', 'A348340', 'A348370', 'A352480', 'A353810', 'A354320', 'A356680', 'A356860', 'A357550', 'A357780', 'A358570', 'A360070', 'A361390', 'A365340', 'A368770', 'A370090', 'A372170', 'A372320', 'A373160', 'A376270', 'A376300', 'A377450', 'A377480', 'A382150', 'A383310', 'A388720', 'A389020', 'A389260', 'A389470', 'A389500', 'A389650', 'A393210', 'A393970', 'A394280', 'A394800', 'A396470', 'A397030', 'A399720', 'A402030', 'A403870', 'A405100', 'A413390', 'A413640', 'A416180', 'A419080', 'A419530', 'A420770', 'A424960', 'A425420', 'A435570', 'A437730', 'A439090', 'A439250', 'A442900', 'A444530', 'A445090', 'A445680', 'A448710', 'A448740', 'A448900', 'A450950', 'A451250', 'A452450', 'A455900', 'A456070', 'A457550', 'A458650', 'A458870', 'A460870', 'A460930', 'A461300', 'A462350', 'A463480', 'A466100', 'A466410', 'A473980', 'A475400', 'A475460', 'A475580', 'A475830', 'A475960', 'A476060', 'A476080', 'A479960', 'A481070', 'A482630', 'A484810', 'A489500']
    objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
    objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")
    k=0
    # 사용 예시
    index_names = get_all_index_names()
    #for name in index_list:
    #    print(name)
    #print(len(index_list))
    for stock_code in codeList[394:]:
        #cnt = 100  # 최근 100일
        #index_names = get_all_index_names()
        #safe_index_list = [name for name in index_list if is_index_calculable(name, objSeries)]
        #print(index_names)
        #series, df = get_ohlcv_series(stock_code, cnt)
        df = pd.read_csv("D:\\졸업프로젝트\\데이터\\" + stock_code + ".csv")
        df = df[::-1].reset_index(drop=True)
        count = len(df)
        if count<60:
            continue
        dates = []
        objSeries = win32com.client.Dispatch("CpIndexes.CpSeries")
        for i in range(count):
            date = df.iloc[i,0]
            open_ = df.iloc[i,1]
            high = df.iloc[i,2]
            low = df.iloc[i,3]
            close = df.iloc[i,4]
            volume = df.iloc[i,5]

            objSeries.Add(close, open_, high, low, volume)
        series=objSeries


        index_columns = calculate_all_indexes(series, index_names)
        # 데이터프레임 결합
        # 유효한 (길이 일치하는) 열만 모아 새로운 딕셔너리로 구성
        valid_columns = {
            key: values for key, values in index_columns.items() if len(values) == len(df)
        }

        # 딕셔너리 → 새로운 DataFrame (인덱스는 기존 df와 동일하게 설정)
        df_new = pd.DataFrame(valid_columns, index=df.index)

        # 기존 df와 수평으로 결합
        df = pd.concat([df, df_new], axis=1)

        # 메모리 조각화 제거 (선택)
        df = df.copy()
        # 결과 확인
        #print(df.tail())
        print(stock_code, k)
        k+=1
        df.to_csv("D:\\졸업프로젝트\\데이터\\C" + stock_code + ".csv", index=False, encoding='utf-8-sig')




#==========================전처리
'''import os
import glob

folder_path = 'C:/example_folder'  # 대상 폴더 경로

# 'CA'로 시작하고 .csv 확장자를 가진 파일만 검색
file_pattern = os.path.join("D:\\졸업프로젝트\\데이터\\", 'CA*.csv')
matching_files = glob.glob(file_pattern)

# 파일 이름만 추출 (전체 경로가 아닌)
file_names = [os.path.basename(f) for f in matching_files]

# 결과 출력
print(file_names)
i=0
# 필요하다면 파일로 저장
with open('CA_csv_filenames.txt', 'w', encoding='utf-8') as f:
    for name in file_names:
        f.write(name + '\n')
        i+=1
print(i)'''
#=========================있는 파일명 끌어모으기 CA기준
import pandas as pd
import numpy as np
from collections import Counter


codeList=['CA000250.csv', 'CA000440.csv', 'CA003100.csv', 'CA003800.csv', 'CA005290.csv', 'CA005670.csv', 'CA007330.csv', 'CA007390.csv', 'CA008830.csv', 'CA009300.csv', 'CA009520.csv', 'CA009780.csv', 'CA011560.csv', 'CA013030.csv', 'CA014620.csv', 'CA017890.csv', 'CA018120.csv', 'CA018290.csv', 'CA020400.csv', 'CA023160.csv', 'CA023900.csv', 'CA023910.csv', 'CA024060.csv', 'CA025770.csv', 'CA025870.csv', 'CA025950.csv', 'CA025980.csv', 'CA028300.csv', 'CA029960.csv', 'CA030520.csv', 'CA031980.csv', 'CA032190.csv', 'CA032300.csv', 'CA032685.csv', 'CA032960.csv', 'CA033100.csv', 'CA033160.csv', 'CA033500.csv', 'CA034950.csv', 'CA035760.csv', 'CA035900.csv', 'CA036190.csv', 'CA036480.csv', 'CA036800.csv', 'CA036810.csv', 'CA036830.csv', 'CA036890.csv', 'CA036930.csv', 'CA037460.csv', 'CA038290.csv', 'CA039030.csv', 'CA039200.csv', 'CA039440.csv', 'CA039610.csv', 'CA039840.csv', 'CA041510.csv', 'CA041830.csv', 'CA042000.csv', 'CA042370.csv', 'CA042420.csv', 'CA042510.csv', 'CA043150.csv', 'CA045100.csv', 'CA045660.csv', 'CA047920.csv', 'CA048410.csv', 'CA048910.csv', 'CA049070.csv', 'CA049430.csv', 'CA049720.csv', 'CA049950.csv', 'CA051160.csv', 'CA051500.csv', 'CA052020.csv', 'CA052400.csv', 'CA053030.csv', 'CA053080.csv', 'CA053160.csv', 'CA053580.csv', 'CA053610.csv', 'CA053800.csv', 'CA054450.csv', 'CA054930.csv', 'CA056190.csv', 'CA058470.csv', 'CA058610.csv', 'CA058970.csv', 'CA059090.csv', 'CA060370.csv', 'CA064290.csv', 'CA064550.csv', 'CA064760.csv', 'CA064820.csv', 'CA064850.csv', 'CA065350.csv', 'CA065660.csv', 'CA065680.csv', 'CA065710.csv', 'CA066620.csv', 'CA067080.csv', 'CA067160.csv', 'CA067280.csv', 'CA067370.csv', 'CA068760.csv', 'CA068930.csv', 'CA069510.csv', 'CA071280.csv', 'CA072020.csv', 'CA072870.csv', 'CA073490.csv', 'CA074600.csv', 'CA078070.csv', 'CA078340.csv', 'CA078600.csv', 'CA079940.csv', 'CA079960.csv', 'CA080010.csv', 'CA080220.csv', 'CA082270.csv', 'CA082920.csv', 'CA083310.csv', 'CA083450.csv', 'CA083650.csv', 'CA083930.csv', 'CA084110.csv', 'CA084370.csv', 'CA085660.csv', 'CA086390.csv', 'CA086450.csv', 'CA086520.csv', 'CA086670.csv', 'CA086900.csv', 'CA087010.csv', 'CA088340.csv', 'CA089010.csv', 'CA089030.csv', 'CA089600.csv', 'CA089970.csv', 'CA089980.csv', 'CA090360.csv', 'CA092130.csv', 'CA092460.csv', 'CA092730.csv', 'CA093320.csv', 'CA093520.csv', 'CA094170.csv', 'CA094360.csv', 'CA094820.csv', 'CA094940.csv', 'CA095340.csv', 'CA095610.csv', 'CA095660.csv', 'CA096240.csv', 'CA096250.csv', 'CA096530.csv', 'CA098070.csv', 'CA098460.csv', 'CA099190.csv', 'CA099320.csv', 'CA099750.csv', 'CA100030.csv', 'CA100120.csv', 'CA101160.csv', 'CA101490.csv', 'CA101930.csv', 'CA101970.csv', 'CA102120.csv', 'CA102710.csv', 'CA102940.csv', 'CA104460.csv', 'CA104830.csv', 'CA106190.csv', 'CA107640.csv', 'CA108380.csv', 'CA108490.csv', 'CA108860.csv', 'CA109080.csv', 'CA109740.csv', 'CA109860.csv', 'CA110990.csv', 'CA112040.csv', 'CA112290.csv', 'CA114840.csv', 'CA115310.csv', 'CA115440.csv', 'CA115450.csv', 'CA115500.csv', 'CA119610.csv', 'CA119850.csv', 'CA120240.csv', 'CA121600.csv', 'CA122640.csv', 'CA122870.csv', 'CA123860.csv', 'CA126340.csv', 'CA126700.csv', 'CA131290.csv', 'CA131970.csv', 'CA136540.csv', 'CA137400.csv', 'CA138610.csv', 'CA140410.csv', 'CA140860.csv', 'CA141080.csv', 'CA143160.csv', 'CA143240.csv', 'CA144510.csv', 'CA145020.csv', 'CA148250.csv', 'CA160190.csv', 'CA160980.csv', 'CA161580.csv', 'CA163280.csv', 'CA166090.csv', 'CA168360.csv', 'CA171090.csv', 'CA173130.csv', 'CA174900.csv', 'CA178320.csv', 'CA179900.csv', 'CA182360.csv', 'CA183300.csv', 'CA186230.csv', 'CA187870.csv', 'CA189300.csv', 'CA190510.csv', 'CA191420.csv', 'CA194700.csv', 'CA195940.csv', 'CA196170.csv', 'CA199800.csv', 'CA200670.csv', 'CA204270.csv', 'CA206650.csv', 'CA211270.csv', 'CA213420.csv', 'CA214150.csv', 'CA214260.csv', 'CA214370.csv', 'CA214430.csv', 'CA214450.csv', 'CA215000.csv', 'CA215200.csv', 'CA215360.csv', 'CA218410.csv', 'CA219130.csv', 'CA220100.csv', 'CA221980.csv', 'CA222160.csv', 'CA222800.csv', 'CA224110.csv', 'CA225570.csv', 'CA226950.csv', 'CA228760.csv', 'CA230240.csv', 'CA232140.csv', 'CA232680.csv', 'CA236200.csv', 'CA237690.csv', 'CA239890.csv', 'CA240810.csv', 'CA241710.csv', 'CA243070.csv', 'CA247540.csv', 'CA251120.csv', 'CA251370.csv', 'CA251970.csv', 'CA253450.csv', 'CA253840.csv', 'CA254490.csv', 'CA256940.csv', 'CA257720.csv', 'CA260970.csv', 'CA263720.csv', 'CA263750.csv', 'CA263860.csv', 'CA264660.csv', 'CA265520.csv', 'CA267980.csv', 'CA270660.csv', 'CA272290.csv', 'CA274090.csv', 'CA277810.csv', 'CA278280.csv', 'CA281740.csv', 'CA282720.csv', 'CA282880.csv', 'CA285490.csv', 'CA289930.csv', 'CA290650.csv', 'CA295310.csv', 'CA298380.csv', 'CA299030.csv', 'CA304100.csv', 'CA304360.csv', 'CA308430.csv', 'CA310210.csv', 'CA314930.csv', 'CA317330.csv', 'CA319660.csv', 'CA323280.csv', 'CA323350.csv', 'CA323990.csv', 'CA328130.csv', 'CA335890.csv', 'CA336570.csv', 'CA336680.csv', 'CA340570.csv', 'CA347850.csv', 'CA348210.csv', 'CA348340.csv', 'CA348370.csv', 'CA352480.csv', 'CA353810.csv', 'CA354320.csv', 'CA356680.csv', 'CA356860.csv', 'CA357550.csv', 'CA357780.csv', 'CA358570.csv', 'CA360070.csv', 'CA361390.csv', 'CA365340.csv', 'CA368770.csv', 'CA370090.csv', 'CA372170.csv', 'CA372320.csv', 'CA376270.csv', 'CA376300.csv', 'CA377450.csv', 'CA377480.csv', 'CA382150.csv', 'CA383310.csv', 'CA388720.csv', 'CA389020.csv', 'CA389260.csv', 'CA389470.csv', 'CA389500.csv', 'CA389650.csv', 'CA393210.csv', 'CA394280.csv', 'CA394800.csv', 'CA396470.csv', 'CA397030.csv', 'CA399720.csv', 'CA402030.csv', 'CA403870.csv', 'CA405100.csv', 'CA413390.csv', 'CA413640.csv', 'CA416180.csv', 'CA419080.csv', 'CA419530.csv', 'CA420770.csv', 'CA424960.csv', 'CA425420.csv', 'CA437730.csv', 'CA439090.csv', 'CA439250.csv', 'CA442900.csv', 'CA445090.csv', 'CA445680.csv', 'CA448710.csv', 'CA448740.csv', 'CA448900.csv', 'CA451250.csv', 'CA455900.csv', 'CA456070.csv', 'CA457550.csv', 'CA458650.csv', 'CA458870.csv', 'CA460930.csv', 'CA461300.csv', 'CA462350.csv', 'CA466100.csv', 'CA466410.csv', 'CA473980.csv', 'CA475400.csv', 'CA475580.csv', 'CA475960.csv', 'CA476080.csv']
k=0
df = pd.read_csv("D:\\졸업프로젝트\\데이터\\ECA457550.csv")
feature_lists = df.columns
    #for name in index_list:
    #    print(name)
    #print(len(index_list))
feature_counter = Counter()
for stock_code in codeList[:]:
    #df = pd.read_csv("D:\\졸업프로젝트\\데이터\\" + stock_code)
    #max_float = 1.7976931348623157e+308
    #df.replace(max_float, np.nan, inplace=True)
    #all_nan_columns = df.columns[df.isna().all()].tolist()
    #print(stock_code, k, all_nan_columns)
    #=========nan변경
    #df.drop(['일목균형표', 'MFI'], axis=1, inplace=True)
    #df = pd.read_csv("D:\\졸업프로젝트\\데이터\\D" + stock_code)
    #===========nan열 삭제
    df = pd.read_csv("D:\\졸업프로젝트\\데이터\\E" + stock_code)
    df['종가_lag1'] = df['종가'].shift(-1)
    corr_matrix = df.corr()
    # 타겟 열이 있다면 (예: 'target'), 그와의 상관계수 절댓값 기준으로 필터링
    target_col = '종가_lag1'  # ← 바꿔줘야 함
    #corr_with_target = corr_matrix[target_col].abs()
    # 상관계수가 0.2 이상인 열만 선택 (기준은 필요에 따라 조정)
    #useful_features = corr_with_target[corr_with_target > 0.2].index.tolist()
    # 유효한 피처만 추출
    #df_filtered = df[useful_features]
    #print(stock_code, k, len(df_filtered.columns), df_filtered.columns.tolist())
    k += 1
    print(k)
    #df.to_csv("D:\\졸업프로젝트\\데이터\\E" + stock_code, index=False, encoding='utf-8-sig')
    if '종가_lag1' in corr_matrix.columns:
        corr_with_target = corr_matrix['종가_lag1'].abs()
        useful_features = corr_with_target[corr_with_target > 0.2].index.tolist()
        feature_counter.update(useful_features)
'''
    # 등장 횟수가 100 이상인 feature만 출력
common_features = [feature for feature, count in feature_counter.items() if count >= 100]

#cnt = 100  # 최근 100일
        #index_names = get_all_index_names()
        #safe_index_list = [name for name in index_list if is_index_calculable(name, objSeries)]
        #print(index_names)
        #series, df = get_ohlcv_series(stock_code, cnt)
print("100회 이상 등장한 유효 feature들:")
print(common_features)
'''
feature_df = pd.DataFrame.from_dict(feature_counter, orient='index', columns=['횟수'])
feature_df = feature_df.sort_values(by='횟수', ascending=False)

# CSV로 저장
feature_df.to_csv("D:\\졸업프로젝트\\데이터\\00feature_count.csv", encoding='utf-8-sig')


'''
common_features=['날짜', '시가', '고가', '저가', '종가', '거래량', '가격& BOX차트', '고저이동평균', '그물망차트', '대표값(Typical Price)', '중간값(Median Price)', '이동평균(라인1개)', '이동평균(라인3개)', '이동평균(수평이동)', '이동평균채널', '표준오차밴드', 'Alligator', 'Bollinger Band', 'DEMA', 'DEMA(simple)', 'Demark', 'Envelope', 'Keltner Channels', 'LRI', 'McGinley Dynamic', 'Parabolic SAR', 'Percent Channel(분/틱용)', 'Pivot Lines', 'Price Channel', 'Projection Bands', 'Starc Bands', 'TEMA', 'TEMA(simple)', 'Tirone Level', 'VIDYA', 'VWMA', 'Weighted Close', 'ZigZag', 'Net Power(4개)', 'BPDL Trend Filter', 'McCellan Summation', 'North Price Action Line', 'On Balance Price', 'Price Change Line', 'TSF', 'Ultimate Oscillator', 'Velocith Index', 'Williams Accumulation Distribution', 'ATR', 'Standard Deviation', 'Standard Error', 'True Range', 'A/D Line', "Bostian's Intraday Intensity Index", 'GM McCellan Summation', 'Morris Intraday Accumulator', 'Negative Volume Index', 'OBV', 'OBV with Average Volume', 'OBV Midpoint', 'Positive Volume Index', 'Price Volume Trend', 'Smooth Accumulation Distribution', 'Special Accumulation Distribution', 'TRIN(inverted)', '종가_lag1', 'Energy', 'Moving Balance Indicator', 'Trend Power Buy', 'Trend Power Sell']
print(len(common_features))'''

#=======================결측치
import pandas as pd
import numpy as np
from collections import Counter


codeList=['CA000250.csv', 'CA000440.csv', 'CA003100.csv', 'CA003800.csv', 'CA005290.csv', 'CA005670.csv', 'CA007330.csv', 'CA007390.csv', 'CA008830.csv', 'CA009300.csv', 'CA009520.csv', 'CA009780.csv', 'CA011560.csv', 'CA013030.csv', 'CA014620.csv', 'CA017890.csv', 'CA018120.csv', 'CA018290.csv', 'CA020400.csv', 'CA023160.csv', 'CA023900.csv', 'CA023910.csv', 'CA024060.csv', 'CA025770.csv', 'CA025870.csv', 'CA025950.csv', 'CA025980.csv', 'CA028300.csv', 'CA029960.csv', 'CA030520.csv', 'CA031980.csv', 'CA032190.csv', 'CA032300.csv', 'CA032685.csv', 'CA032960.csv', 'CA033100.csv', 'CA033160.csv', 'CA033500.csv', 'CA034950.csv', 'CA035760.csv', 'CA035900.csv', 'CA036190.csv', 'CA036480.csv', 'CA036800.csv', 'CA036810.csv', 'CA036830.csv', 'CA036890.csv', 'CA036930.csv', 'CA037460.csv', 'CA038290.csv', 'CA039030.csv', 'CA039200.csv', 'CA039440.csv', 'CA039610.csv', 'CA039840.csv', 'CA041510.csv', 'CA041830.csv', 'CA042000.csv', 'CA042370.csv', 'CA042420.csv', 'CA042510.csv', 'CA043150.csv', 'CA045100.csv', 'CA045660.csv', 'CA047920.csv', 'CA048410.csv', 'CA048910.csv', 'CA049070.csv', 'CA049430.csv', 'CA049720.csv', 'CA049950.csv', 'CA051160.csv', 'CA051500.csv', 'CA052020.csv', 'CA052400.csv', 'CA053030.csv', 'CA053080.csv', 'CA053160.csv', 'CA053580.csv', 'CA053610.csv', 'CA053800.csv', 'CA054450.csv', 'CA054930.csv', 'CA056190.csv', 'CA058470.csv', 'CA058610.csv', 'CA058970.csv', 'CA059090.csv', 'CA060370.csv', 'CA064290.csv', 'CA064550.csv', 'CA064760.csv', 'CA064820.csv', 'CA064850.csv', 'CA065350.csv', 'CA065660.csv', 'CA065680.csv', 'CA065710.csv', 'CA066620.csv', 'CA067080.csv', 'CA067160.csv', 'CA067280.csv', 'CA067370.csv', 'CA068760.csv', 'CA068930.csv', 'CA069510.csv', 'CA071280.csv', 'CA072020.csv', 'CA072870.csv', 'CA073490.csv', 'CA074600.csv', 'CA078070.csv', 'CA078340.csv', 'CA078600.csv', 'CA079940.csv', 'CA079960.csv', 'CA080010.csv', 'CA080220.csv', 'CA082270.csv', 'CA082920.csv', 'CA083310.csv', 'CA083450.csv', 'CA083650.csv', 'CA083930.csv', 'CA084110.csv', 'CA084370.csv', 'CA085660.csv', 'CA086390.csv', 'CA086450.csv', 'CA086520.csv', 'CA086670.csv', 'CA086900.csv', 'CA087010.csv', 'CA088340.csv', 'CA089010.csv', 'CA089030.csv', 'CA089600.csv', 'CA089970.csv', 'CA089980.csv', 'CA090360.csv', 'CA092130.csv', 'CA092460.csv', 'CA092730.csv', 'CA093320.csv', 'CA093520.csv', 'CA094170.csv', 'CA094360.csv', 'CA094820.csv', 'CA094940.csv', 'CA095340.csv', 'CA095610.csv', 'CA095660.csv', 'CA096240.csv', 'CA096250.csv', 'CA096530.csv', 'CA098070.csv', 'CA098460.csv', 'CA099190.csv', 'CA099320.csv', 'CA099750.csv', 'CA100030.csv', 'CA100120.csv', 'CA101160.csv', 'CA101490.csv', 'CA101930.csv', 'CA101970.csv', 'CA102120.csv', 'CA102710.csv', 'CA102940.csv', 'CA104460.csv', 'CA104830.csv', 'CA106190.csv', 'CA107640.csv', 'CA108380.csv', 'CA108490.csv', 'CA108860.csv', 'CA109080.csv', 'CA109740.csv', 'CA109860.csv', 'CA110990.csv', 'CA112040.csv', 'CA112290.csv', 'CA114840.csv', 'CA115310.csv', 'CA115440.csv', 'CA115450.csv', 'CA115500.csv', 'CA119610.csv', 'CA119850.csv', 'CA120240.csv', 'CA121600.csv', 'CA122640.csv', 'CA122870.csv', 'CA123860.csv', 'CA126340.csv', 'CA126700.csv', 'CA131290.csv', 'CA131970.csv', 'CA136540.csv', 'CA137400.csv', 'CA138610.csv', 'CA140410.csv', 'CA140860.csv', 'CA141080.csv', 'CA143160.csv', 'CA143240.csv', 'CA144510.csv', 'CA145020.csv', 'CA148250.csv', 'CA160190.csv', 'CA160980.csv', 'CA161580.csv', 'CA163280.csv', 'CA166090.csv', 'CA168360.csv', 'CA171090.csv', 'CA173130.csv', 'CA174900.csv', 'CA178320.csv', 'CA179900.csv', 'CA182360.csv', 'CA183300.csv', 'CA186230.csv', 'CA187870.csv', 'CA189300.csv', 'CA190510.csv', 'CA191420.csv', 'CA194700.csv', 'CA195940.csv', 'CA196170.csv', 'CA199800.csv', 'CA200670.csv', 'CA204270.csv', 'CA206650.csv', 'CA211270.csv', 'CA213420.csv', 'CA214150.csv', 'CA214260.csv', 'CA214370.csv', 'CA214430.csv', 'CA214450.csv', 'CA215000.csv', 'CA215200.csv', 'CA215360.csv', 'CA218410.csv', 'CA219130.csv', 'CA220100.csv', 'CA221980.csv', 'CA222160.csv', 'CA222800.csv', 'CA224110.csv', 'CA225570.csv', 'CA226950.csv', 'CA228760.csv', 'CA230240.csv', 'CA232140.csv', 'CA232680.csv', 'CA236200.csv', 'CA237690.csv', 'CA239890.csv', 'CA240810.csv', 'CA241710.csv', 'CA243070.csv', 'CA247540.csv', 'CA251120.csv', 'CA251370.csv', 'CA251970.csv', 'CA253450.csv', 'CA253840.csv', 'CA254490.csv', 'CA256940.csv', 'CA257720.csv', 'CA260970.csv', 'CA263720.csv', 'CA263750.csv', 'CA263860.csv', 'CA264660.csv', 'CA265520.csv', 'CA267980.csv', 'CA270660.csv', 'CA272290.csv', 'CA274090.csv', 'CA277810.csv', 'CA278280.csv', 'CA281740.csv', 'CA282720.csv', 'CA282880.csv', 'CA285490.csv', 'CA289930.csv', 'CA290650.csv', 'CA295310.csv', 'CA298380.csv', 'CA299030.csv', 'CA304100.csv', 'CA304360.csv', 'CA308430.csv', 'CA310210.csv', 'CA314930.csv', 'CA317330.csv', 'CA319660.csv', 'CA323280.csv', 'CA323350.csv', 'CA323990.csv', 'CA328130.csv', 'CA335890.csv', 'CA336570.csv', 'CA336680.csv', 'CA340570.csv', 'CA347850.csv', 'CA348210.csv', 'CA348340.csv', 'CA348370.csv', 'CA352480.csv', 'CA353810.csv', 'CA354320.csv', 'CA356680.csv', 'CA356860.csv', 'CA357550.csv', 'CA357780.csv', 'CA358570.csv', 'CA360070.csv', 'CA361390.csv', 'CA365340.csv', 'CA368770.csv', 'CA370090.csv', 'CA372170.csv', 'CA372320.csv', 'CA376270.csv', 'CA376300.csv', 'CA377450.csv', 'CA377480.csv', 'CA382150.csv', 'CA383310.csv', 'CA388720.csv', 'CA389020.csv', 'CA389260.csv', 'CA389470.csv', 'CA389500.csv', 'CA389650.csv', 'CA393210.csv', 'CA394280.csv', 'CA394800.csv', 'CA396470.csv', 'CA397030.csv', 'CA399720.csv', 'CA402030.csv', 'CA403870.csv', 'CA405100.csv', 'CA413390.csv', 'CA413640.csv', 'CA416180.csv', 'CA419080.csv', 'CA419530.csv', 'CA420770.csv', 'CA424960.csv', 'CA425420.csv', 'CA437730.csv', 'CA439090.csv', 'CA439250.csv', 'CA442900.csv', 'CA445090.csv', 'CA445680.csv', 'CA448710.csv', 'CA448740.csv', 'CA448900.csv', 'CA451250.csv', 'CA455900.csv', 'CA456070.csv', 'CA457550.csv', 'CA458650.csv', 'CA458870.csv', 'CA460930.csv', 'CA461300.csv', 'CA462350.csv', 'CA466100.csv', 'CA466410.csv', 'CA473980.csv', 'CA475400.csv', 'CA475580.csv', 'CA475960.csv', 'CA476080.csv']
k=0
df = pd.read_csv("D:\\졸업프로젝트\\데이터\\00feature_count.csv")
first_column_list = df.iloc[1:70, 0].tolist()
print(first_column_list)
for stock_code in codeList[:]:
    k += 1
    print(k, stock_code)
    try:
        df = pd.read_csv("D:\\졸업프로젝트\\데이터\\E" + stock_code)
        df=df[first_column_list]
        df = df.iloc[120:].reset_index(drop=True)
        #inf가 남는데 얘도 고쳐줘
        df_filled = np.where(np.isfinite(df), df, 0)  # 결측치 제거
        df_filled = np.clip(df, -1e6, 1e6).astype(np.float32)  # 이상치도 제거 + 형변환
        df_filled.to_csv("D:\\졸업프로젝트\\데이터\\F" + stock_code, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"{stock_code} 처리 중 오류 발생: {e}")
        continue

#==============npz
import pandas as pd
import numpy as np
from collections import Counter
import os


codeList=['CA000250.csv', 'CA000440.csv', 'CA003100.csv', 'CA003800.csv', 'CA005290.csv', 'CA005670.csv', 'CA007330.csv', 'CA007390.csv', 'CA008830.csv', 'CA009300.csv', 'CA009520.csv', 'CA009780.csv', 'CA011560.csv', 'CA013030.csv', 'CA014620.csv', 'CA017890.csv', 'CA018120.csv', 'CA018290.csv', 'CA020400.csv', 'CA023160.csv', 'CA023900.csv', 'CA023910.csv', 'CA024060.csv', 'CA025770.csv', 'CA025870.csv', 'CA025950.csv', 'CA025980.csv', 'CA028300.csv', 'CA029960.csv', 'CA030520.csv', 'CA031980.csv', 'CA032190.csv', 'CA032300.csv', 'CA032685.csv', 'CA032960.csv', 'CA033100.csv', 'CA033160.csv', 'CA033500.csv', 'CA034950.csv', 'CA035760.csv', 'CA035900.csv', 'CA036190.csv', 'CA036480.csv', 'CA036800.csv', 'CA036810.csv', 'CA036830.csv', 'CA036890.csv', 'CA036930.csv', 'CA037460.csv', 'CA038290.csv', 'CA039030.csv', 'CA039200.csv', 'CA039440.csv', 'CA039610.csv', 'CA039840.csv', 'CA041510.csv', 'CA041830.csv', 'CA042000.csv', 'CA042370.csv', 'CA042420.csv', 'CA042510.csv', 'CA043150.csv', 'CA045100.csv', 'CA045660.csv', 'CA047920.csv', 'CA048410.csv', 'CA048910.csv', 'CA049070.csv', 'CA049430.csv', 'CA049720.csv', 'CA049950.csv', 'CA051160.csv', 'CA051500.csv', 'CA052020.csv', 'CA052400.csv', 'CA053030.csv', 'CA053080.csv', 'CA053160.csv', 'CA053580.csv', 'CA053610.csv', 'CA053800.csv', 'CA054450.csv', 'CA054930.csv', 'CA056190.csv', 'CA058470.csv', 'CA058610.csv', 'CA058970.csv', 'CA059090.csv', 'CA060370.csv', 'CA064290.csv', 'CA064550.csv', 'CA064760.csv', 'CA064820.csv', 'CA064850.csv', 'CA065350.csv', 'CA065660.csv', 'CA065680.csv', 'CA065710.csv', 'CA066620.csv', 'CA067080.csv', 'CA067160.csv', 'CA067280.csv', 'CA067370.csv', 'CA068760.csv', 'CA068930.csv', 'CA069510.csv', 'CA071280.csv', 'CA072020.csv', 'CA072870.csv', 'CA073490.csv', 'CA074600.csv', 'CA078070.csv', 'CA078340.csv', 'CA078600.csv', 'CA079940.csv', 'CA079960.csv', 'CA080010.csv', 'CA080220.csv', 'CA082270.csv', 'CA082920.csv', 'CA083310.csv', 'CA083450.csv', 'CA083650.csv', 'CA083930.csv', 'CA084110.csv', 'CA084370.csv', 'CA085660.csv', 'CA086390.csv', 'CA086450.csv', 'CA086520.csv', 'CA086670.csv', 'CA086900.csv', 'CA087010.csv', 'CA088340.csv', 'CA089010.csv', 'CA089030.csv', 'CA089600.csv', 'CA089970.csv', 'CA089980.csv', 'CA090360.csv', 'CA092130.csv', 'CA092460.csv', 'CA092730.csv', 'CA093320.csv', 'CA093520.csv', 'CA094170.csv', 'CA094360.csv', 'CA094820.csv', 'CA094940.csv', 'CA095340.csv', 'CA095610.csv', 'CA095660.csv', 'CA096240.csv', 'CA096250.csv', 'CA096530.csv', 'CA098070.csv', 'CA098460.csv', 'CA099190.csv', 'CA099320.csv', 'CA099750.csv', 'CA100030.csv', 'CA100120.csv', 'CA101160.csv', 'CA101490.csv', 'CA101930.csv', 'CA101970.csv', 'CA102120.csv', 'CA102710.csv', 'CA102940.csv', 'CA104460.csv', 'CA104830.csv', 'CA106190.csv', 'CA107640.csv', 'CA108380.csv', 'CA108490.csv', 'CA108860.csv', 'CA109080.csv', 'CA109740.csv', 'CA109860.csv', 'CA110990.csv', 'CA112040.csv', 'CA112290.csv', 'CA114840.csv', 'CA115310.csv', 'CA115440.csv', 'CA115450.csv', 'CA115500.csv', 'CA119610.csv', 'CA119850.csv', 'CA120240.csv', 'CA121600.csv', 'CA122640.csv', 'CA122870.csv', 'CA123860.csv', 'CA126340.csv', 'CA126700.csv', 'CA131290.csv', 'CA131970.csv', 'CA136540.csv', 'CA137400.csv', 'CA138610.csv', 'CA140410.csv', 'CA140860.csv', 'CA141080.csv', 'CA143160.csv', 'CA143240.csv', 'CA144510.csv', 'CA145020.csv', 'CA148250.csv', 'CA160190.csv', 'CA160980.csv', 'CA161580.csv', 'CA163280.csv', 'CA166090.csv', 'CA168360.csv', 'CA171090.csv', 'CA173130.csv', 'CA174900.csv', 'CA178320.csv', 'CA179900.csv', 'CA182360.csv', 'CA183300.csv', 'CA186230.csv', 'CA187870.csv', 'CA189300.csv', 'CA190510.csv', 'CA191420.csv', 'CA194700.csv', 'CA195940.csv', 'CA196170.csv', 'CA199800.csv', 'CA200670.csv', 'CA204270.csv', 'CA206650.csv', 'CA211270.csv', 'CA213420.csv', 'CA214150.csv', 'CA214260.csv', 'CA214370.csv', 'CA214430.csv', 'CA214450.csv', 'CA215000.csv', 'CA215200.csv', 'CA215360.csv', 'CA218410.csv', 'CA219130.csv', 'CA220100.csv', 'CA221980.csv', 'CA222160.csv', 'CA222800.csv', 'CA224110.csv', 'CA225570.csv', 'CA226950.csv', 'CA228760.csv', 'CA230240.csv', 'CA232140.csv', 'CA232680.csv', 'CA236200.csv', 'CA237690.csv', 'CA239890.csv', 'CA240810.csv', 'CA241710.csv', 'CA243070.csv', 'CA247540.csv', 'CA251120.csv', 'CA251370.csv', 'CA251970.csv', 'CA253450.csv', 'CA253840.csv', 'CA254490.csv', 'CA256940.csv', 'CA257720.csv', 'CA260970.csv', 'CA263720.csv', 'CA263750.csv', 'CA263860.csv', 'CA264660.csv', 'CA265520.csv', 'CA267980.csv', 'CA270660.csv', 'CA272290.csv', 'CA274090.csv', 'CA277810.csv', 'CA278280.csv', 'CA281740.csv', 'CA282720.csv', 'CA282880.csv', 'CA285490.csv', 'CA289930.csv', 'CA290650.csv', 'CA295310.csv', 'CA298380.csv', 'CA299030.csv', 'CA304100.csv', 'CA304360.csv', 'CA308430.csv', 'CA310210.csv', 'CA314930.csv', 'CA317330.csv', 'CA319660.csv', 'CA323280.csv', 'CA323350.csv', 'CA323990.csv', 'CA328130.csv', 'CA335890.csv', 'CA336570.csv', 'CA336680.csv', 'CA340570.csv', 'CA347850.csv', 'CA348210.csv', 'CA348340.csv', 'CA348370.csv', 'CA352480.csv', 'CA353810.csv', 'CA354320.csv', 'CA356680.csv', 'CA356860.csv', 'CA357550.csv', 'CA357780.csv', 'CA358570.csv', 'CA360070.csv', 'CA361390.csv', 'CA365340.csv', 'CA368770.csv', 'CA370090.csv', 'CA372170.csv', 'CA372320.csv', 'CA376270.csv', 'CA376300.csv', 'CA377450.csv', 'CA377480.csv', 'CA382150.csv', 'CA383310.csv', 'CA388720.csv', 'CA389020.csv', 'CA389260.csv', 'CA389470.csv', 'CA389500.csv', 'CA389650.csv', 'CA393210.csv', 'CA394280.csv', 'CA394800.csv', 'CA396470.csv', 'CA397030.csv', 'CA399720.csv', 'CA402030.csv', 'CA403870.csv', 'CA405100.csv', 'CA413390.csv', 'CA413640.csv', 'CA416180.csv', 'CA419080.csv', 'CA419530.csv', 'CA420770.csv', 'CA424960.csv', 'CA425420.csv', 'CA437730.csv', 'CA439090.csv', 'CA439250.csv', 'CA442900.csv', 'CA445090.csv', 'CA445680.csv', 'CA448710.csv', 'CA448740.csv', 'CA448900.csv', 'CA451250.csv', 'CA455900.csv', 'CA456070.csv', 'CA457550.csv', 'CA458650.csv', 'CA458870.csv', 'CA460930.csv', 'CA461300.csv', 'CA462350.csv', 'CA466100.csv', 'CA466410.csv', 'CA473980.csv', 'CA475400.csv', 'CA475580.csv', 'CA475960.csv', 'CA476080.csv']
k=0
#df = pd.read_csv("D:\\졸업프로젝트\\데이터\\00feature_count.csv")
#first_column_list = df.iloc[1:70, 0].tolist()
#print(first_column_list)
for stock_code in codeList[:]:
    k += 1
    print(k, stock_code)
    try:
        df = pd.read_csv("D:\\졸업프로젝트\\데이터\\F" + stock_code)
        df = df.drop(columns=['날짜'])
        if len(df) < 61:
            continue
        windows = []
        for i in range(min(340,len(df)-60)):  # 마지막 인덱스는 len(df)-60 + 1
            window = df.iloc[i:i + 61].to_numpy()  # (60, n_features)
            windows.append(window)
        result = np.array(windows)
        np.savez("D:\\졸업프로젝트\\학습데이터\\test\\"+f"{stock_code}.npz", data=result)
        print(result.shape)
        #test
        if len(df)<340+60:
            continue
        windows = []
        for i in range(340,min(290+340,len(df)-60)):  # 마지막 인덱스는 len(df)-60 + 1
            window = df.iloc[i:i + 61].to_numpy()  # (60, n_features)
            windows.append(window)
        result = np.array(windows)
        np.savez("D:\\졸업프로젝트\\학습데이터\\validation\\" + f"{stock_code}.npz", data=result)
        print(result.shape)
        #validation
        if len(df)<340+240+60:
            continue
        windows = []
        for i in range(290+340,len(df) - 60):  # 마지막 인덱스는 len(df)-60 + 1
            window = df.iloc[i:i + 61].to_numpy()  # (60, n_features)
            windows.append(window)
        result = np.array(windows)
        np.savez("D:\\졸업프로젝트\\학습데이터\\train\\" + f"{stock_code}.npz", data=result)
        print(result.shape)
        #train
    except Exception as e:
        print(f"{stock_code} 처리 중 오류 발생: {e}")
        continue