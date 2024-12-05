from langchain.tools import tool
from langchain_gigachat.chat_models import GigaChat
from pydantic import BaseModel, Field
import requests
import os
from utils.data_caching import load_from_cache, save_to_cache
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Класс результата OSM
class OSMResult(BaseModel):
    name: str = Field(description="Название места")
    address: str = Field(description="Адрес места")
    latitude: float = Field(description="Широта")
    longitude: float = Field(description="Долгота")
    nearby: list[str] = Field(description="Список мест поблизости")
    tags: dict = Field(default_factory=dict, description="Теги объекта")
    message: str = Field(description="Сообщение о результатах поиска")

# Класс результата WikiData
class WikiDataResult(BaseModel):
    title: str = Field(description="Заголовок статьи")
    description: str = Field(description="Описание статьи")
    extract: str = Field(description="Краткое содержание статьи")
    url: str = Field(description="Ссылка на статью")
    message: str = Field(description="Сообщение о результатах поиска")

class OpenTripMapResult(BaseModel):
    places: list[dict] = Field(description="Список мест интереса поблизости")
    message: str = Field(description="Сообщение о результатах поиска")

@tool
def get_osm_data(place: str) -> OSMResult:
    """Получить информацию о месте и близлежащих объектах из OSM."""
    cache_key = f"osm_{place}"
    cached_data = load_from_cache(cache_key)
    if cached_data:
        print('Using cached_data')
        return OSMResult(**cached_data)

    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place,
        "format": "json",
        "addressdetails": 1,
        "limit": 1
    }
    headers = {"User-Agent": "YourAppName/1.0 (zabydy40@gmail.com)"}

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data:
            return OSMResult(name="", address="", latitude=0, longitude=0, nearby=[],tags={},
                             message=f"Место '{place}' не найдено.")

        place_data = data[0]
        name = place_data.get('name', place)
        latitude = float(place_data['lat'])
        longitude = float(place_data['lon'])
        address = place_data.get('display_name', 'Адрес не найден')
        tags = place_data.get('extratags', {})  # Получаем теги объекта

        # Найти близлежащие объекты
        nearby_url = "https://overpass-api.de/api/interpreter"
        nearby_query = f"""
        [out:json];
        node(around:1000,{latitude},{longitude});
        out;
        """

        nearby_response = requests.post(nearby_url, data=nearby_query)
        nearby_response.raise_for_status()
        nearby_data = nearby_response.json()
        nearby_places = [element['tags'].get('name', 'Неизвестное место')
                         for element in nearby_data.get('elements', [])
                         if 'tags' in element and 'name' in element['tags']
                         ]
        print('==========================================')
        print('osm_data')
        print(place)
        print('')
        print(data)
        print('')
        print('nearby_places')
        print(nearby_places)
        print('==========================================')
        result = OSMResult(
            name=name,
            address=address,
            latitude=latitude,
            longitude=longitude,
            nearby=nearby_places[:7],  # Только 5 мест
            tags=tags,
            message=f"Информация о месте '{place}' успешно получена."
        )
        save_to_cache(cache_key, result.model_dump())
        return result

    except requests.exceptions.RequestException as e:
        return OSMResult(name="", address="", latitude=0, longitude=0, nearby=[],tags={},
                         message=f"Ошибка при запросе OSM: {str(e)}")

@tool
def get_wikidata_info(query: str) -> WikiDataResult:
    """Получить информацию о достопримечательности из WikiData, Wikimedia и Wikipedia."""
    cache_key = f"wikidata_{query}"
    cached_data = load_from_cache(cache_key)
    if cached_data:
        return WikiDataResult(**cached_data)

    try:
        # Поиск на WikiData
        wikidata_url = "https://www.wikidata.org/w/api.php"
        wikidata_params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json"
        }
        wikidata_response = requests.get(wikidata_url, params=wikidata_params)
        wikidata_response.raise_for_status()
        wikidata_data = wikidata_response.json()
        print('==========================================')
        print('wikidata_data')
        print('')
        print(wikidata_data)
        print('==========================================')
        print('')
        if wikidata_data.get("search"):
            wikidata_result = wikidata_data["search"][0]
            title = wikidata_result.get("label", "Название не найдено")
            description = wikidata_result.get("description", "Описание не найдено")
            wikidata_url = wikidata_result.get("concepturi", "URL не найден")
        else:
            title, description, wikidata_url = "", "", ""

        # Поиск на Wikipedia
        wikipedia_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        wikipedia_response = requests.get(wikipedia_url)
        wikipedia_response.raise_for_status()
        wikipedia_data = wikipedia_response.json()
        print('==========================================')
        print('wikipedia_data')
        print('')
        print(wikipedia_data)
        print('==========================================')
        print('')
        extract = wikipedia_data.get('extract', 'Краткое содержание не найдено')
        wiki_url = wikipedia_data.get('content_urls', {}).get('desktop', {}).get('page', 'URL не найден')

        result = WikiDataResult(
            title=title or wikipedia_data.get('title', 'Название не найдено'),
            description=description or wikipedia_data.get('description', 'Описание не найдено'),
            extract=extract,
            url=wikidata_url or wiki_url,
            message=f"Информация о '{query}' успешно получена."
        )
        save_to_cache(cache_key, result.model_dump())
        return result

    except requests.exceptions.RequestException as e:
        return WikiDataResult(title="", description="", extract="", url="",
                              message=f"Ошибка при запросе данных: {str(e)}")

@tool
def get_opentripmap_info(place: str, radius: int = 1000) -> OpenTripMapResult:
    """Получить информацию о ближайших местах интереса из OpenTripMap."""
    # Используем OSM для получения координат места
    osm_result = get_osm_data(place)
    if not osm_result.latitude or not osm_result.longitude:
        return OpenTripMapResult(
            places=[],
            message=f"Не удалось найти координаты для места '{place}' через OSM."
        )

    cache_key = f"opentripmap_{place}_{radius}"
    cached_data = load_from_cache(cache_key)
    if cached_data:
        return OpenTripMapResult(**cached_data)

    base_url = "https://api.opentripmap.com/0.1/en/places"
    api_key = '5ae2e3f221c38a28845f05b64d31947c669d2f87e5a4e8fd485debd7'
    headers = {"User-Agent": "YourAppName/1.0 (zabydy40@gmail.com)"}

    # Получение мест интереса в радиусе
    place_url = f"{base_url}/radius"
    place_params = {"radius": radius, "lon": osm_result.longitude, "lat": osm_result.latitude, "apikey": api_key}
    try:
        place_response = requests.get(place_url, params=place_params, headers=headers)
        place_response.raise_for_status()
        place_data = place_response.json()
        print('==========================================')
        print('OpenTripMapResult')
        print('')
        print(place_data)
        print('==========================================')
        print('')
        if not place_data.get("features"):
            return OpenTripMapResult(
                places=[],
                message=f"Места интереса в радиусе {radius} метров от '{place}' не найдены."
            )

        # Обработка результатов
        places = []
        for feature in place_data["features"]:
            places.append({
                "name": feature["properties"].get("name", "Без названия"),
                "description": feature["properties"].get("kinds", "Описание отсутствует"),
                "latitude": feature["geometry"].get("coordinates", [0, 0])[1],
                "longitude": feature["geometry"].get("coordinates", [0, 0])[0],
                "url": feature["properties"].get("url", "")
            })

        result = OpenTripMapResult(
            places=places,
            message=f"Найдено {len(places)} мест(а) интереса в радиусе {radius} метров от '{place}'."
        )

        save_to_cache(cache_key, result.model_dump())
        return result

    except requests.exceptions.RequestException as e:
        return OpenTripMapResult(
            places=[],
            message=f"Ошибка при запросе OpenTripMap: {str(e)}"
        )


def filter_nearby_places(nearby_places: list[str], preferences: list[str]) -> list[str]:
    """Фильтрует места по интересам пользователя."""
    filtered_places = []
    for place in nearby_places:
        if any(preference.lower() in place.lower() for preference in preferences):
            filtered_places.append(place)
    return filtered_places
# Функция для получения полной информации о месте и его окрестностях
def fetch_full_info(main_place: str, preferences: list[str]):
    # Получаем информацию о главном месте
    main_info = get_osm_data(main_place)
    if not main_info.name:
        return {"main": main_info, "desc": WikiDataResult(title="", description="Информация о месте отсутствует, или место не существует", extract="", url="",
                              message="Незнакомое место"), "nearby": []}

    # Получаем описание из WikiData
    wiki_info = get_wikidata_info(main_info.name)

    # Фильтруем ближайшие места по предпочтениям
    filtered_nearby_places = filter_nearby_places(main_info.nearby, preferences)

    # Собираем информацию о каждом месте
    nearby_info = []
    for place in filtered_nearby_places:
        osm_data = get_osm_data(place)
        wiki_data = get_wikidata_info(place)
        nearby_info.append({
            "osm": osm_data,
            "wiki": wiki_data
        })

    return {"main": main_info, "desc": wiki_info, "nearby": nearby_info}


# Агент для формирования экскурсии
# class OpenMapAndWikiAgent:
#     def __init__(self, llm, agent_functions, system_prompt):
#         self.llm = llm
#         self.agent_functions = agent_functions
#         self.system_prompt = system_prompt
#
#     def run(self, user_input: str, preferences: list[str]):
#         tools = self.agent_functions
#         agent = create_react_agent(self.llm, tools, state_modifier=self.system_prompt)
#
#         # Получаем данные для экскурсии с учетом предпочтений
#         data = fetch_full_info(user_input, preferences)
#         print(data)
#
#         # Формируем сообщение для LLM
#         input_message = f"""
#         Место: {data['main'].name}
#         Адрес: {data['main'].address}
#         Описание: {data['desc'].description}
#         Информация: {data['desc'].extract}
#
#         Популярные места поблизости:
#         {"; ".join([f"{place['osm'].name} - {place['wiki'].description}. {place['wiki'].extract}" for place in data['nearby']])}
#
#         Answer language: Русский
#         """
#         print('INPUT: ', input_message)
#         print('')
#
#         try:
#             result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
#             return result["messages"][-1].content
#         except Exception as e:
#             return f"Error: {str(e)}"
#
# # Пример использования
# if __name__ == "__main__":
#     # Инициализация LLM (например, GigaChat)
#     API_TOKEN = ""  # Укажите токен API
#     MODEL_NAME = "GigaChat"
#     SYSTEM_PROMPT = "Твоя профессия - экскурсовод. Твоя задача, рассказать интересный, познавательный и достоверный рассказ про указанное пользователем место! Основывай свои ответы на информации из tools. Если запрос пользователя не связан с тематикой экскурсий, не отвечай ему"
#     # Инициализация клиентов и модели
#     model = GigaChat(
#         credentials=API_TOKEN,
#         scope="GIGACHAT_API_PERS",
#         model=MODEL_NAME,
#         verify_ssl_certs=False
#     )
#
#     # Создание агента
#     agent = OpenMapAndWikiAgent(
#         llm=model,
#         # agent_functions=[get_osm_data, get_wikidata_info],
#         agent_functions=[],
#         system_prompt=SYSTEM_PROMPT
#     )
#
#     user_preferences = ["museum"]
#     # Запрос
#     user_query = "Гонконг"
#     print('ANSWER: ', agent.run(user_query, preferences=user_preferences))

# === Улучшенная логика агента ===
class InteractiveTourGuide:
    def __init__(self, llm, tools, system_prompt):
        self.agent = create_react_agent(llm, tools, state_modifier=system_prompt)
        self.chat_history = []

    def run(self, user_input: str, preferences: list[str]):
        """Обрабатывает пользовательский запрос и формирует экскурсию."""
        self.chat_history.append(HumanMessage(content=f"Место: {user_input}. Предпочтения: {preferences}"))
        try:
            result = self.agent.invoke({"messages": self.chat_history})
            self.chat_history.append(result["messages"][-1])
            return result["messages"][-1].content
        except Exception as e:
            return f"Error: {str(e)}"

# === Основной скрипт ===
if __name__ == "__main__":
    # Инициализация LLM (например, GigaChat)
    API_TOKEN = ""  # Укажите токен API
    MODEL_NAME = "GigaChat"
    SYSTEM_PROMPT = """Ты экскурсовод.
    Твоя задача, рассказать интересный, познавательный и достоверный рассказ про указанное пользователем место!
    Составляй маршруты и рассказывай истории. Учитывай контекст диалога.
    Основывай свои ответы на информации из tools.
    Если запрос пользователя не связан с тематикой экскурсий, не отвечай ему"""
    # Инициализация клиентов и модели
    model = GigaChat(
        credentials=API_TOKEN,
        scope="GIGACHAT_API_PERS",
        model=MODEL_NAME,
        verify_ssl_certs=False
    )
    tour_guide = InteractiveTourGuide(llm=model, tools=[get_osm_data, get_wikidata_info, get_opentripmap_info], system_prompt=SYSTEM_PROMPT)

    print("Добро пожаловать в интерактивный гид! Напишите 'выход', чтобы завершить.")
    user_preferences = input("Введите ваши предпочтения (через запятую): ").split(",")
    while True:
        user_query = input("Введите место или запрос: ")
        if user_query.lower() == "выход":
            print("Спасибо за использование гида!")
            break
        response = tour_guide.run(user_query, user_preferences)
        print(response)
