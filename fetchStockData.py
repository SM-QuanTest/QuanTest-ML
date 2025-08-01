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
