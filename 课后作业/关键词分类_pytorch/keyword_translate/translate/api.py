import time
import requests

key = 'AIzaSyBmNYceXws8PIztEO5NTseytNSeJiR06Z0'
time_interval = 3
max_retry = 3


def translate(text, src_language, target_language, retry=max_retry):
    is_array = isinstance(text, list)
    url = f"https://translation.googleapis.com/language/translate/v2?key={key}"

    data = {
        "q": text,
        "format": "text",
        "source": src_language,
        "target": target_language
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }

    try:
        response = requests.request("POST", url, headers=headers, json=data)
        result = response.json()
        if is_array:
            return [item['translatedText'] for item in result['data']['translations']]
        else:
            return result['data']['translations'][0]['translatedText']
    except:
        print(f"Failed to translate: {text}")
        if retry > 0:
            delay = time_interval * (max_retry - retry + 1)
            print(
                f"Retry after {delay} seconds")
            time.sleep(delay)
            return translate(text, src_language, target_language, retry - 1)

        return None
