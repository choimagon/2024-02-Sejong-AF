import os
import json
import pandas as pd

def extract_to_csv(json_file_path, output_csv_path):
    data = []

    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        # JSON 데이터 내의 "data" 리스트에서 필요한 필드 추출
        for item in json_data["data"]:
            source_original = item.get("source_cleaned", "")
            mtpe = item.get("MTPE", "")
            
            data.append({"source_cleaned": source_original, "MTPE": mtpe})

    # DataFrame으로 변환 후 CSV로 저장
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

# JSON 파일과 출력 CSV 파일 경로 설정
json_file_path = '/Users/choejihun/HakSul/Sample_trans/sample/E_EA_10001.json'
output_csv_path = 'data/sample_trans.csv'

extract_to_csv(json_file_path, output_csv_path)
