import requests
import time

def lat_lon(address, api_key):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}

    try:
        res = requests.get(url, headers=headers, params=params)
        if res.status_code != 200:
            print(f"[{res.status_code}] 요청 실패: {address}")
            return None, None

        result = res.json()
        documents = result.get('documents', [])
        
        if len(documents) == 0:
            print(f"위경도 없음: {address}")
            return None, None

        doc = documents[0]
        return doc['y'], doc['x']

    except Exception as e:
        print(f"예외 발생 - {address}: {e}")
        return None, None