#json파일 -> csv파일로 바꿔주는 코드
import os
import json
import pandas as pd

def extract_to_csv(folder_path, output_csv_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                
                paragraph = json_data.get("paragraph", "")
                gen_summary = json_data.get("gen_summary", "")
                
                data.append({"paragraph": paragraph, "gen_summary": gen_summary})

    df = pd.DataFrame(data)
    
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

folder_path = '/Users/choejihun/HakSul/Sample_summary/02.라벨링데이터/그룹1/19'
output_csv_path = 'data/sample.csv'

extract_to_csv(folder_path, output_csv_path)

output_csv_path
