## T5 트랜스포머 모델임 <br>
번역과 요약에 좀 특화됬다고함. <br>

깃 클론해서 가져가세요.<br>
> ```git clone https://github.com/choimagon/2024-02-Sejong-AF.git``` <br>

## 데이터셋  <br>
데이터셋 (법률 데이터 요약) : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71794   <br>
-> 법률 데이터 요약을 사용함으로써 내용 요약에 좀 더 신뢰도 높을 거 같음. <br>
데이터셋 (국제 학술대회용 전문분야 한영/영한 통번역 데이터) : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71693 <br>
-> 전문용어가 많이 나오는 데이터가 좀 더 신뢰도 있을거같음. <br>

ex) 요약 데이터셋 (json파일인데 -> 이거 csv 바꿔서, data파일에 넣어둠)
```
{
  "id": "1037_1958",
  "bill_id": "PRC_R1Y5O0C1R2X8H1H7R5M3Z0F3G7X4Y1",
  "title": "국회법 일부개정법률안(김춘진의원 등 10인)",
  "committee": "국회운영위원회",
  "field": "국회법",
  "enactment": "N",
  "amendment": "Y",
  "proposer": "김춘진의원 등 10인 ",
  "advisor": "한공식",
  "date": "2016-05-29",
  "session": "19",
  "paragraph": "현행 국회법 제85조의3은 예산안의 법정기한 내 처리를 위하여 도입된 것으로 예산안등과 세입예산안 부수 법률안의 위원회 심사를 11월 30일까지 마치지 못한 경우, 그 다음 날 본회의에 부의된 것으로 간주하도록 하고 있음.\r\n그러나 작년 12월 2015회계연도 예산안 심사과정에서 세입예산안 부수 법률안 지정과 관련하여, 세입예산안과 관련된 국민건강증진부담금 인상에 관한 사항뿐만 아니라 세입예산안과 관련이 없는 흡연경고 그림 부착에 관한 내용까지 담고 있는 하나의 “국민건강증진법 개정안”을 세입예산안 부수 법률안으로 지정함에 따라 이에 대한 적절성 여부가 논란이 된바 있었음.\r\n이 과정에서 세입예산안과 직접적인 관련이 없는 내용도 세입예산안 부수 법률안으로 지정되어 위원회의 심사 및 의결없이 본회의에 자동으로 부의되도록 하는 것은 예산안의 법정기한 내 처리라는 국회법 제85조의3의 입법목적 달성을 위하여 예산안등과 직접적인 관련이 없는 내용에까지도 위원회의 법안심사 권한을 과도하게 제한하는 결과를 초래하는 측면이 있음.\r\n이에 개정안은 법안 발의시 세입예산안에 “직접적인 영향”을 미치는 경우에 한하여 세입예산안 부수 법률안 여부를 표시하도록 하고, 의장이 세입예산안 부수 법률안을 지정함에 있어 세입예산안과 무관하게 특정한 행정 목적을 위하여 국민의 권리를 제한하거나 의무를 부과하는 조항이 포함된 경우, 이를 세입예산안 부수 법률안으로 지정하지 못하게 함으로써, 세입예산안 부수법률안의 지정요건을 예산안과 직접적인 관련성이 있는 법률안으로 제한하려는 것으로 보임.",
  "ext_summary": "현행 국회법 제85조의3은 예산안의 법정기한 내 처리를 위하여 도입된 것으로 예산안등과 세입예산안 부수 법률안의 위원회 심사를 11월 30일까지 마치지 못한 경우, 그 다음 날 본회의에 부의된 것으로 간주하도록 하고 있음.\r\n이에 개정안은 법안 발의시 세입예산안에 “직접적인 영향”을 미치는 경우에 한하여 세입예산안 부수 법률안 여부를 표시하도록 하고, 의장이 세입예산안 부수 법률안을 지정함에 있어 세입예산안과 무관하게 특정한 행정 목적을 위하여 국민의 권리를 제한하거나 의무를 부과하는 조항이 포함된 경우, 이를 세입예산안 부수 법률안으로 지정하지 못하게 함으로써, 세입예산안 부수법률안의 지정요건을 예산안과 직접적인 관련성이 있는 법률안으로 제한하려는 것으로 보임.",
  "gen_summary": "이에 개정안은 법안 발의시 의장이 세입예산안 부수 법률안을 지정함에 있어 세입예산안과 무관하게 특정한 행정 목적을 위하여 국민의 권리를 제한하거나 의무를 부과하는 조항이 포함된 경우, 지정하지 못하게 함으로써, 세입예산안 부수법률안의 지정요건을 예산안과 직접적인 관련성이 있는 법률안으로 제한하려는 것으로 보임.",
  "terminology": "법안 발의, 세입예산안 부수 법률안, 행정 목적, 예산안, 의장",
  "disposal": "임기만료폐기"
}

```

ex) 번역 데이터셋 (json파일인데 -> 이거 csv 바꿔서, data파일에 넣어둠)
```
{
    "sn": "EKEA100000002",
    "file_name": "E_EA_10001.wav",
    "file_format": ".wav",
    "audio_duration": "00:14:12.533",
    "audio_start": "00:00:05.283",
    "audio_end": "00:00:15.622",
    "conf_name": null,
    "domain": "공학",
    "subdomain": "토목공학",
    "info_place": null,
    "info_date": "2023",
    "info_gender": "male",
    "info_age": 50,
    "source_original": "-",
    "source_cleaned": "Interdisciplinarity: Is there room for It in undergraduate engineering education's futures?",
    "source_language": "en",
    "target_language": "ko",
    "MT": "학제간: 학부 공학 교육의 미래에 그것이 들어갈 여지가 있습니까?",
    "MTPE": "학제간 연구: 학부 공학 교육의 미래에 학제간 연구의 여지가 존재하는가.",
    "final_audio_duration": "00:12:37.095",
    "license": true
},
```

## 각 파일 설명 <br>
> 폴더 : data <br>
> ```data :  학습에 필요한 데이터셋을 csv로 변환해둠(현재 sample만 csv로 만들어둠)``` <br>
> 원본 data는 용량 문제로 업로드 안됌 <br>


>  python 파일 : mainT5.py, classT5.py, useClassT5.py, csvView.py, trans-json2csv.py, trans-Trans_json2csv.py <br>
>  ```mainT5.py : 학습할때 사용하는 T5 모델 코드``` <br>
>  ```classT5.py : 실제 학습된 T5모델을 가져오기위해 만든 class(그냥 클래스 선언만 있는거)``` <br>
>  ```useClassT5.py : classT5를 가져와서 실제 사용방법을 보여주는 코드(학습된 가중치들과 편향들을 용량이 커서 업로드 안됌)``` <br>
>  ```csvView.py : csv파일의 일부를 보여주는 코드(그냥 궁금하면 확인하는 용도)```<br>
>  ```trans-json2csv.py,  trans-Trans_json2csv.p: 학습 데이터의 원본은 json 파일인데 필요한 부분만 가져와서 csv파일로 변환해주는 파일```<br>

번역 모델 : 90개의 데이터로 40번 학습 돌렸을때의 결과 (나쁘지않은데 아마 과적합일거임->본 학술제로 들어가는 순간 제대로 학습해야징)
![image](https://github.com/user-attachments/assets/22a5b23e-3232-4294-9def-18f97164cc2f)

요약 모델 
![image](https://github.com/user-attachments/assets/faa3a5dc-b37e-444a-a560-0f900b03fcfa)


