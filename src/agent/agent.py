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
            return OSMResult(name="", address="", latitude=0, longitude=0, nearby=[],
                             message=f"Место '{place}' не найдено.")

        place_data = data[0]
        name = place_data.get('name', place)
        latitude = float(place_data['lat'])
        longitude = float(place_data['lon'])
        address = place_data.get('display_name', 'Адрес не найден')

        # Найти близлежащие объекты
        nearby_url = "https://overpass-api.de/api/interpreter"
        nearby_query = f"""
        [out:json];
        node(around:1000,{latitude},{longitude})[tourism~"attraction|museum|viewpoint"];
        out;
        """

        nearby_response = requests.post(nearby_url, data=nearby_query)
        nearby_response.raise_for_status()
        nearby_data = nearby_response.json()

        nearby_places = [element['tags'].get('name', 'Неизвестное место')
                         for element in nearby_data.get('elements', [])
                         if 'tags' in element and 'name' in element['tags']]

        result = OSMResult(
            name=name,
            address=address,
            latitude=latitude,
            longitude=longitude,
            nearby=nearby_places[:7],  # Только 5 мест
            message=f"Информация о месте '{place}' успешно получена."
        )
        save_to_cache(cache_key, result.model_dump())
        print('get_osm_data: ', result.model_dump())
        return result

    except requests.exceptions.RequestException as e:
        return OSMResult(name="", address="", latitude=0, longitude=0, nearby=[],
                         message=f"Ошибка при запросе OSM: {str(e)}")

# Класс результата WikiData
class WikiDataResult(BaseModel):
    title: str = Field(description="Заголовок статьи")
    description: str = Field(description="Описание статьи")
    extract: str = Field(description="Краткое содержание статьи")
    url: str = Field(description="Ссылка на статью")
    message: str = Field(description="Сообщение о результатах поиска")

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

# Функция для получения полной информации о месте и его окрестностях
def fetch_full_info(main_place: str):
    # Получаем информацию о главном месте
    main_info = get_osm_data(main_place)
    # Если не удалось найти место, возвращаем только его сообщение
    if not main_info.name:
        return {"main": main_info, "desc": WikiDataResult(title="", description="Информация о месте отсутствует, или место не существует", extract="", url="",
                              message="Незнакомое место"), "nearby": []}
    wiki_info = get_wikidata_info(main_info.name)

    # Получаем информацию о близлежащих местах
    nearby_info = []
    for place in main_info.nearby:
        osm_data = get_osm_data(place)
        wiki_data = get_wikidata_info(place)
        nearby_info.append({
            "osm": osm_data,
            "wiki": wiki_data
        })

    return {"main": main_info, "desc": wiki_info, "nearby": nearby_info}

# Агент для формирования экскурсии
class OpenMapAndWikiAgent:
    def __init__(self, llm, agent_functions, system_prompt):
        self.llm = llm
        self.agent_functions = agent_functions
        self.system_prompt = system_prompt

    def run(self, user_input: str):
        tools = self.agent_functions
        agent = create_react_agent(self.llm, tools, state_modifier=self.system_prompt)

        # Получаем данные для экскурсии
        data = fetch_full_info(user_input)
        print(data)

        # Формируем сообщение для LLM
        input_message = f"""
        Место: {data['main'].name}
        Адрес: {data['main'].address}
        Описание: {data['desc'].description}
        Информация: {data['desc'].extract}

        Популярные места поблизости:
        {"; ".join([f"{place['osm'].name} - {place['wiki'].description}. {place['wiki'].extract}" for place in data['nearby']])}
        
        Answer language: Русский
        """
        print('INPUT: ', input_message)
        print('')
        try:
            result = agent.invoke({"messages": [HumanMessage(content=input_message)]})
            return result["messages"][-1].content
        except Exception as e:
            return f"Error: {str(e)}"


# Пример использования
if __name__ == "__main__":
    # Инициализация LLM (например, GigaChat)
    API_TOKEN = ""  # Укажите токен API
    MODEL_NAME = "GigaChat"
    SYSTEM_PROMPT = "Твоя профессия - экскурсовод. Твоя задача, рассказать интересный, познавательный и достоверный рассказ про указанное пользователем место! Основывай свои ответы на информации из tools. Если запрос пользователя не связан с тематикой экскурсий, не отвечай ему"
    # Инициализация клиентов и модели
    model = GigaChat(
        credentials=API_TOKEN,
        scope="GIGACHAT_API_PERS",
        model=MODEL_NAME,
        verify_ssl_certs=False
    )

    # Создание агента
    agent = OpenMapAndWikiAgent(
        llm=model,
        # agent_functions=[get_osm_data, get_wikidata_info],
        agent_functions=[],
        system_prompt=SYSTEM_PROMPT
    )

    # Запрос
    user_query = "Страна Незнания"
    print('ANSWER: ', agent.run(user_query))