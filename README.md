## T5 트랜스포머 모델임 <br>
번역과 요약에 좀 특화됬다고함. <br>

깃 클론해서 가져가세요.<br>
```git clone https://github.com/choimagon/2024-02-Sejong-AF.git```

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
