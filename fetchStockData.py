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
                    print(
                        f"{index_name:<25} | 조건1: {term1:<3} 조건2: {term2:<3} 조건3: {term3:<3} 조건4: {term4:<3} Signal: {signal}")
                except Exception as e:
                    print(f"⚠️ '{index_name}' 지표 조건 불러오기 실패: {e}")
        except Exception as e:
            print(f"⚠️ 카테고리 {category} 지표 리스트 불러오기 실패: {e}")


if __name__ == "__main__":
    objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
    codes = get_kosdaq_stock_codes()
    print(codes)
    print(len(codes))
    i = 0

    finalcode=[]

    for code in codes[:]:
        capVal = get_market_cap(code)  # 시가총액 받아와서 필터링
        i += 1
        print(i)
        if capVal < 5000:
            print("skip: " + code)
            continue
        data = fetch_historical_data_with_delay(code, 20150414, 20250414)
        columns = ["날짜", "시가", "고가", "저가", "종가", "거래량"]
        df = pd.DataFrame(data, columns=columns)
        df["종가"] = pd.to_numeric(df["종가"], errors='coerce')
        if df.empty or df["종가"].dropna().empty:
            print("❌ Skip (종가 없음 또는 모두 NaN):", code)
            continue
        current_price = df["종가"].iloc[-1]
        if current_price == 0:
            print("❌ Skip (종가 0):", code)
            continue

        print("✅ PASS:", code)
        finalcode.append(code)
        df.to_csv("D:\\졸업프로젝트\\CEEMD\\" + code + ".csv", index=False, encoding='utf-8-sig')

        time.sleep(3.6)
    print(finalcode)
    df = pd.DataFrame(finalcode, columns=['code'])
    df.to_csv("D:\\졸업프로젝트\\CEEMD\\filteredCode.csv", index=False, encoding='utf-8-sig')




''' #윗부분 전부 실행 후 아래 코드 실행하면 대신증권 로그인 없이 파일만으로도 계산가능!
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
# 실행 예시
if __name__ == "__main__":
    #codeList부분은 달라진다면 가진 파일을 폴더에서 끌어오는 걸로 변경해야 해
    codeList=['A000250', 'A000440', 'A003100', 'A003800', 'A005290', 'A005670', 'A007330', 'A007390', 'A008830', 'A009300', 'A009520', 'A009780', 'A011560', 'A013030', 'A014620', 'A017890', 'A018120', 'A018290', 'A020400', 'A023160', 'A023900', 'A023910', 'A024060', 'A025770', 'A025870', 'A025950', 'A025980', 'A028300', 'A029960', 'A030520', 'A031980', 'A032190', 'A032300', 'A032685', 'A032960', 'A033100', 'A033160', 'A033500', 'A034950', 'A035760', 'A035900', 'A036190', 'A036480', 'A036800', 'A036810', 'A036830', 'A036890', 'A036930', 'A037460', 'A038290', 'A039030', 'A039200', 'A039440', 'A039610', 'A039840', 'A041510', 'A041830', 'A042000', 'A042370', 'A042420', 'A042510', 'A043150', 'A045100', 'A045660', 'A047920', 'A048410', 'A048910', 'A049070', 'A049430', 'A049720', 'A049950', 'A051160', 'A051500', 'A052020', 'A052400', 'A053030', 'A053080', 'A053160', 'A053580', 'A053610', 'A053800', 'A054450', 'A054930', 'A056190', 'A058470', 'A058610', 'A058970', 'A059090', 'A060370', 'A064290', 'A064550', 'A064760', 'A064820', 'A064850', 'A065350', 'A065660', 'A065680', 'A065710', 'A066620', 'A067080', 'A067160', 'A067280', 'A067370', 'A068760', 'A068930', 'A069510', 'A071280', 'A072020', 'A072870', 'A073490', 'A074600', 'A078070', 'A078340', 'A078600', 'A079940', 'A079960', 'A080010', 'A080220','A082270', 'A082920', 'A083310', 'A083450', 'A083650', 'A083930', 'A084110', 'A084370', 'A085660', 'A086390', 'A086450', 'A086520', 'A086670', 'A086900', 'A087010', 'A088340', 'A089010', 'A089030', 'A089600', 'A089970', 'A089980', 'A090360', 'A092130', 'A092460', 'A092730', 'A093320', 'A093520', 'A094170', 'A094360', 'A094820', 'A094940', 'A095340', 'A095610', 'A095660', 'A096240', 'A096250', 'A096530', 'A098070', 'A098460', 'A099190', 'A099320', 'A099750', 'A100030', 'A100120', 'A101160', 'A101490', 'A101930', 'A101970', 'A102120', 'A102710', 'A102940', 'A104460', 'A104830', 'A106190', 'A107640', 'A108380', 'A108490', 'A108860', 'A109080', 'A109740', 'A109860', 'A110990', 'A112040', 'A112290', 'A114840', 'A115310', 'A115440', 'A115450', 'A115500', 'A119610', 'A119850', 'A120240', 'A121600', 'A122640', 'A122870', 'A123860', 'A126340', 'A126700', 'A131290', 'A131970', 'A136540', 'A137400', 'A138610', 'A140410', 'A140860', 'A141080', 'A143160', 'A143240', 'A144510', 'A145020', 'A148250', 'A160190', 'A160980', 'A161580', 'A163280', 'A166090', 'A168360', 'A171090', 'A173130', 'A174900', 'A178320', 'A179900', 'A182360', 'A183300', 'A186230', 'A187870', 'A189300', 'A190510', 'A191420', 'A194700', 'A195940', 'A196170', 'A199800', 'A200670', 'A204270', 'A206650', 'A211270', 'A213420', 'A214150', 'A214260', 'A214370', 'A214430', 'A214450', 'A215000', 'A215200', 'A215360', 'A218410', 'A219130', 'A220100', 'A221980', 'A222160', 'A222800', 'A224110', 'A225570', 'A226590', 'A226950', 'A228760', 'A230240', 'A232140', 'A232680', 'A236200', 'A237690', 'A239890', 'A240550', 'A240810', 'A241710', 'A243070', 'A247540', 'A251120', 'A251370', 'A251970', 'A253450', 'A253840', 'A254490', 'A256940', 'A257720', 'A260970', 'A263720', 'A263750', 'A263860', 'A264660', 'A265520', 'A267980', 'A270660', 'A272290', 'A274090', 'A277810', 'A278280', 'A281740', 'A282720', 'A282880', 'A285490', 'A289930', 'A290650', 'A295310', 'A298380', 'A299030', 'A303810', 'A304100', 'A304360', 'A308430', 'A310210', 'A314930', 'A317330', 'A319660', 'A323280', 'A323350', 'A323990', 'A328130', 'A335890', 'A336570', 'A336680', 'A340570', 'A347850', 'A348210', 'A348340', 'A348370', 'A352480', 'A353810', 'A354320', 'A356680', 'A356860', 'A357550', 'A357780', 'A358570', 'A360070', 'A361390', 'A365340', 'A368770', 'A370090', 'A372170', 'A372320', 'A373160', 'A376270', 'A376300', 'A377450', 'A377480', 'A382150', 'A383310', 'A388720', 'A389020', 'A389260', 'A389470', 'A389500', 'A389650', 'A393210', 'A393970', 'A394280', 'A394800', 'A396470', 'A397030', 'A399720', 'A402030', 'A403870', 'A405100', 'A413390', 'A413640', 'A416180', 'A419080', 'A419530', 'A420770', 'A424960', 'A425420', 'A435570', 'A437730', 'A439090', 'A439250', 'A442900', 'A444530', 'A445090', 'A445680', 'A448710', 'A448740', 'A448900', 'A450950', 'A451250', 'A452450', 'A455900', 'A456070', 'A457550', 'A458650', 'A458870', 'A460870', 'A460930', 'A461300', 'A462350', 'A463480', 'A466100', 'A466410', 'A473980', 'A475400', 'A475460', 'A475580', 'A475830', 'A475960', 'A476060', 'A476080', 'A479960', 'A481070', 'A482630', 'A484810', 'A489500']
    objStockChart = win32com.client.Dispatch("CpSysDib.StockChart")
    objIndex = win32com.client.Dispatch("CpIndexes.CpIndex")
    k=0
    # 사용 예시
    index_names = get_all_index_names()
    #for name in index_list:
    #    print(name)
    #print(len(index_list))
    for stock_code in codeList[:]:
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

'''
