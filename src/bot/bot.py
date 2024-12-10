from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
import asyncio
import aiohttp
from config import *

# Токен бота
#  t.me/ultrapro14maxtripbot
BOT_TOKEN = os.getenv('BOT_TOKEN')

# Инициализация бота и диспетчера
bot = Bot(BOT_TOKEN)
dp = Dispatcher()

# URL для запроса
url = "http://0.0.0.0:8000/generate_text"

# Асинхронная функция для выполнения POST-запроса
async def post_request(url, data):
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(url, json=data) as response:
                # Если ответ не успешен, выбрасываем исключение
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        # Обработка ошибки сети или API
        return {"text": "Sorry, I couldn't get an answer from the server. Please try again later."}

# Обработчик команды '/start'
@dp.message(Command('start'))
async def start(message: types.Message):
    await message.reply("Welcome! How can I help you today?")

# Обработчик для получения сообщений и отправки их на сервер
@dp.message()
async def answer(message: types.Message):
    data = {"question": message.text}  # Формируем данные для запроса
    received_answer = asyncio.create_task(post_request(url, data))  # Отправляем запрос асинхронно
    response_json = await received_answer  # Ожидаем ответа от сервера
    answer_text = response_json.get("text", "Sorry, something went wrong.")
    await message.reply(answer_text)  # Отправляем полученный ответ

# Основная асинхронная функция для старта бота
async def main():
    await dp.start_polling(bot)

# Запуск бота
if __name__ == "__main__":
    asyncio.run(main())
