import aiohttp
import asyncio


async def get_temperature_async(city, api_key):
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

            if response.status == 200:
                temperature = data['main']['temp']
                return temperature
            else:
                return {
                    "cod": response.status,
                    "message": data.get('message', 'Неизвестная ошибка')
                }