#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the validation/test-only GLM-5.2 quality challenge supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from audit_split_isolation import (
    PRODUCTION_NEAR_DUPLICATE_THRESHOLD,
    audit_rows,
)
from build_quality_dataset import DEFAULT_SYSTEM, read_jsonl, validate_rows
from generate_full_quality_outputs_sglang import evaluation_cluster_id

DATASET = "project-authored/glm52-heldout-quality-challenge"
DATASET_REVISION = "heldout-quality-challenge-v1"
REVIEWER = "project-reference-structure-audit-v1"
MINIMUM_SLICE_ROWS = 10
MINIMUM_SLICE_CLUSTERS = 5
CLEAN_VIEW_REVISION = "mixture_targeted_wikipedia_v4_train_1792"
CLEAN_TRAIN_ROWS_SHA256 = "3131c44ae33051ee8a8fc1ae91e9c8aa94a2750fbe89e0307bb3bd3ff46a262e"
INLINE_SOURCE_DATASETS = (
    DATASET,
    "project-authored/glm52-targeted-quality",
    "wikimedia/wikipedia",
)
EXPECTED_FAMILIES = (
    "markdown-list",
    "markdown-table",
    "markdown-code",
    "markdown-mixed",
    "general-russian",
    "accidental-han-spontaneous",
    "han-cleanup",
    "chinese-retention",
)

LIST_CASES = (
    ("Осмотр метеостанции", ("проверить крепления", "снять показания", "записать время")),
    ("Приём музейного экспоната", ("сверить номер", "осмотреть упаковку", "подписать акт")),
    ("Подготовка школьной экскурсии", ("уточнить маршрут", "собрать согласия", "назначить сопровождающих")),
    ("Проверка аптечки", ("сверить перечень", "заменить просроченное", "опломбировать футляр")),
    ("Настройка аудиогида", ("загрузить дорожки", "проверить громкость", "включить навигацию")),
    ("Учёт библиотечного фонда", ("принять издания", "нанести шифры", "обновить каталог")),
    ("Запуск теплицы", ("проверить полив", "настроить освещение", "записать температуру")),
    ("Подготовка веломаршрута", ("осмотреть покрытие", "отметить ремонт", "обновить карту")),
    ("Архивирование интервью", ("проверить звук", "добавить метаданные", "создать резервную копию")),
    ("Осмотр учебной лаборатории", ("пересчитать приборы", "проверить питание", "закрыть журнал")),
    ("Подготовка выставки минералов", ("сверить этикетки", "настроить свет", "проверить витрины")),
    ("Обход лесной тропы", ("осмотреть указатели", "убрать препятствия", "отметить координаты")),
    ("Приём театрального реквизита", ("сверить ведомость", "проверить состояние", "разместить на складе")),
    ("Подготовка радиопередачи", ("согласовать темы", "проверить микрофоны", "записать заставку")),
    ("Проверка макета журнала", ("сверить оглавление", "проверить подписи", "утвердить обложку")),
    ("Настройка планетария", ("загрузить сценарий", "свести проекторы", "проверить затемнение")),
    ("Учёт спортивного инвентаря", ("пересчитать комплекты", "отложить повреждённое", "обновить карточки")),
    ("Подготовка полевой кухни", ("проверить горелки", "заполнить ёмкости", "разложить посуду")),
    ("Осмотр фотолаборатории", ("проверить реактивы", "настроить таймер", "закрыть светозащиту")),
    ("Организация книжной ярмарки", ("разметить стенды", "проверить списки", "установить навигацию")),
)

TABLE_CASES = (
    ("учёт приливов", ("Высота", "2,4 м"), ("Время", "06:40"), ("Пост", "Северный")),
    ("наблюдение за ульями", ("Семья", "№ 7"), ("Температура", "+31 °C"), ("Осмотр", "завершён")),
    ("контроль киноплёнки", ("Катушка", "К-18"), ("Длина", "620 м"), ("Состояние", "без разрывов")),
    ("проверка водомера", ("Секция", "В-3"), ("Показание", "184,2"), ("Пломба", "целая")),
    ("учёт семян", ("Партия", "Л-42"), ("Масса", "18 кг"), ("Влажность", "9 %")),
    ("осмотр маяка", ("Лампа", "рабочая"), ("Батарея", "87 %"), ("Связь", "устойчивая")),
    ("приём керамики", ("Ящик", "12"), ("Предметов", "36"), ("Сколы", "не найдены")),
    ("замер снежного покрова", ("Маршрут", "Ю-5"), ("Глубина", "47 см"), ("Ветер", "слабый")),
    ("контроль сцены", ("Занавес", "исправен"), ("Свет", "проверен"), ("Выходы", "свободны")),
    ("инвентаризация карт", ("Шкаф", "А-9"), ("Листов", "128"), ("Опись", "обновлена")),
    ("проверка обсерватории", ("Купол", "открывается"), ("Оптика", "чистая"), ("Погода", "ясно")),
    ("учёт костюмов", ("Стойка", "Р-4"), ("Комплектов", "23"), ("Ремонт", "2 позиции")),
    ("осмотр причала", ("Секция", "П-6"), ("Крепления", "затянуты"), ("Освещение", "работает")),
    ("контроль гербария", ("Папка", "Г-15"), ("Листов", "54"), ("Влажность", "норма")),
    ("приём декораций", ("Сцена", "вторая"), ("Панелей", "11"), ("Маркировка", "нанесена")),
    ("проверка локомакета", ("Контур", "внешний"), ("Стрелки", "8"), ("Питание", "12 В")),
    ("учёт проб воды", ("Станция", "Р-2"), ("Проб", "16"), ("Хранение", "+4 °C")),
    ("осмотр колокольни", ("Ярус", "третий"), ("Крепёж", "без люфта"), ("Доступ", "закрыт")),
    ("контроль типографии", ("Тираж", "750"), ("Листы", "А3"), ("Цвет", "согласован")),
    ("проверка радиорубки", ("Канал", "резервный"), ("Сигнал", "−62 дБм"), ("Журнал", "заполнен")),
)

CODE_CASES = (
    (
        "clamp_level",
        "ограничивает уровень диапазоном от нуля до ста",
        "def clamp_level(value):\n    return max(0, min(100, value))",
    ),
    ("is_even", "проверяет чётность целого числа", "def is_even(value):\n    return value % 2 == 0"),
    ("meters_to_cm", "переводит метры в сантиметры", "def meters_to_cm(value):\n    return value * 100"),
    (
        "first_or_none",
        "возвращает первый элемент либо None",
        "def first_or_none(items):\n    return next(iter(items), None)",
    ),
    (
        "normalize_space",
        "схлопывает повторные пробелы",
        'def normalize_space(text):\n    return " ".join(text.split())',
    ),
    (
        "safe_ratio",
        "возвращает ноль при нулевом знаменателе",
        "def safe_ratio(a, b):\n    return 0.0 if b == 0 else a / b",
    ),
    (
        "unique_sorted",
        "удаляет повторы и сортирует значения",
        "def unique_sorted(items):\n    return sorted(set(items))",
    ),
    (
        "has_prefix",
        "проверяет наличие заданного префикса",
        "def has_prefix(text, prefix):\n    return text.startswith(prefix)",
    ),
    ("minutes_to_seconds", "переводит минуты в секунды", "def minutes_to_seconds(value):\n    return value * 60"),
    (
        "positive_only",
        "оставляет только положительные числа",
        "def positive_only(values):\n    return [value for value in values if value > 0]",
    ),
    (
        "mean_or_zero",
        "считает среднее либо возвращает ноль",
        "def mean_or_zero(values):\n    return sum(values) / len(values) if values else 0.0",
    ),
    (
        "ends_with_dot",
        "проверяет точку в конце строки",
        'def ends_with_dot(text):\n    return text.rstrip().endswith(".")',
    ),
    ("join_labels", "соединяет подписи через запятую", 'def join_labels(labels):\n    return ", ".join(labels)'),
    (
        "nonempty",
        "отбрасывает пустые строки",
        "def nonempty(values):\n    return [value for value in values if value.strip()]",
    ),
    ("square", "возводит число в квадрат", "def square(value):\n    return value * value"),
    (
        "is_inside",
        "проверяет принадлежность числа закрытому интервалу",
        "def is_inside(value, low, high):\n    return low <= value <= high",
    ),
    (
        "last_or_none",
        "возвращает последний элемент либо None",
        "def last_or_none(items):\n    return items[-1] if items else None",
    ),
    (
        "count_true",
        "считает истинные значения",
        "def count_true(values):\n    return sum(bool(value) for value in values)",
    ),
    (
        "strip_suffix",
        "удаляет известный суффикс",
        "def strip_suffix(text, suffix):\n    return text.removesuffix(suffix)",
    ),
    (
        "bounded_add",
        "складывает числа и ограничивает сумму сверху",
        "def bounded_add(a, b, limit):\n    return min(a + b, limit)",
    ),
)

MIXED_CASES = (
    (
        "Паспорт акустического зала",
        ("вместимость — 240 мест", "реверберация — 1,4 секунды", "рядов — 12"),
        "замеры выполнены без зрителей",
        "https://example.test/acoustics",
    ),
    (
        "Карточка ботанической оранжереи",
        ("секция — тропическая", "влажность — 78 %", "полив — утром"),
        "датчики проверены вручную",
        "https://example.test/greenhouse",
    ),
    (
        "Сводка реставрационной мастерской",
        ("объект — рама", "слоёв грунта — три", "этап — укрепление"),
        "фотографии состояния сохранены",
        "https://example.test/restoration",
    ),
    (
        "Памятка волонтёра фестиваля",
        ("вход — восточный", "смена — четыре часа", "рация — канал 6"),
        "координатор доступен у сцены",
        "https://example.test/volunteer",
    ),
    (
        "Описание маршрута катера",
        ("отправление — 09:20", "остановок — две", "причал — № 4"),
        "жилеты находятся под сиденьями",
        "https://example.test/boat",
    ),
    (
        "Карточка читального зала",
        ("мест — 48", "тишина — обязательна", "выдача — до 19:30"),
        "редкие издания читают только в зале",
        "https://example.test/reading-room",
    ),
    (
        "Справка по гончарной печи",
        ("объём — 120 литров", "пик — 1180 °C", "остывание — 9 часов"),
        "дверцу открывают после полного охлаждения",
        "https://example.test/kiln",
    ),
    (
        "Карточка походного лагеря",
        ("палаток — семь", "вода — у штаба", "отбой — 23:00"),
        "костры разрешены только на площадке",
        "https://example.test/camp",
    ),
    (
        "Сводка студии звукозаписи",
        ("микрофонов — пять", "частота — 48 кГц", "дублей — три"),
        "исходные дорожки не нормализованы",
        "https://example.test/studio",
    ),
    (
        "Паспорт пешеходного моста",
        ("длина — 86 метров", "опор — четыре", "осмотр — ежемесячно"),
        "нагрузка ограничена указателями",
        "https://example.test/bridge",
    ),
    (
        "Карточка минералогической коллекции",
        ("образцов — 315", "шкафов — восемь", "каталог — цифровой"),
        "новые образцы помещены в карантин",
        "https://example.test/minerals",
    ),
    (
        "Сводка репетиционного класса",
        ("пюпитров — 22", "роялей — один", "занятие — 90 минут"),
        "окна закрывают перед записью",
        "https://example.test/rehearsal",
    ),
    (
        "Памятка смотрителя башни",
        ("ступеней — 146", "площадок — три", "вход — по билетам"),
        "верхний ярус закрывают при ветре",
        "https://example.test/tower",
    ),
    (
        "Паспорт учебного полигона",
        ("секторов — шесть", "групп — четыре", "сеанс — 45 минут"),
        "защитное снаряжение проверяют до входа",
        "https://example.test/training-ground",
    ),
    (
        "Карточка мастерской печати",
        ("станков — два", "красок — четыре", "бумага — хлопковая"),
        "пробный оттиск утверждает художник",
        "https://example.test/printshop",
    ),
    (
        "Сводка орнитологического поста",
        ("наблюдателей — трое", "смена — шесть часов", "оптика — 10×42"),
        "координаты гнёзд не публикуются",
        "https://example.test/birds",
    ),
    (
        "Описание ледовой площадки",
        ("длина — 56 метров", "сеансов — пять", "заточка — на месте"),
        "лёд обновляют после третьего сеанса",
        "https://example.test/ice",
    ),
    (
        "Карточка археологического квадрата",
        ("сторона — пять метров", "слой — второй", "находок — 17"),
        "каждая находка получает полевой номер",
        "https://example.test/dig",
    ),
    (
        "Сводка радиолюбительской станции",
        ("диапазонов — три", "антенн — две", "позывной — РМ7К"),
        "сеанс связи внесён в журнал",
        "https://example.test/radio",
    ),
    (
        "Памятка для планетарного показа",
        ("сеанс — 38 минут", "мест — 96", "яркость — средняя"),
        "двери закрывают до запуска проекции",
        "https://example.test/planetarium",
    ),
)

GENERAL_RUSSIAN_CASES = (
    (
        "Почему иней чаще появляется ясной ночью?",
        "Ясной ночью поверхность быстро отдаёт тепло излучением, охлаждается ниже точки замерзания, и водяной пар оседает на ней кристаллами льда.",
    ),
    (
        "Чем эскиз отличается от чертежа?",
        "Эскиз свободно передаёт замысел и пропорции, а чертёж фиксирует форму и размеры по установленным правилам.",
    ),
    (
        "Зачем перед концертом настраивают инструменты?",
        "Настройка выравнивает высоту звуков, поэтому инструменты согласованно звучат друг с другом во время исполнения.",
    ),
    (
        "Почему бумажные книги хранят вдали от сырости?",
        "Избыточная влажность деформирует бумагу, ослабляет переплёт и создаёт условия для появления плесени.",
    ),
    (
        "Что показывает масштаб на карте?",
        "Масштаб связывает расстояние на карте с соответствующим расстоянием на местности.",
    ),
    (
        "Почему семенам нужен покой перед посевом?",
        "Период покоя помогает семенам пережить неблагоприятное время и начать прорастание при подходящих условиях.",
    ),
    (
        "Для чего в музее регулируют освещение?",
        "Умеренный свет позволяет рассмотреть экспонаты и одновременно уменьшает выцветание чувствительных материалов.",
    ),
    (
        "Почему мосты снабжают деформационными швами?",
        "Швы дают конструкции безопасно расширяться и сжиматься при изменении температуры.",
    ),
    (
        "Чем наблюдение отличается от измерения?",
        "Наблюдение описывает явление, а измерение сопоставляет его характеристику с принятой единицей и даёт числовой результат.",
    ),
    (
        "Зачем аудиозапись сохраняют в исходном формате?",
        "Исходный файл сохраняет максимум доступных данных и остаётся надёжной основой для последующей обработки.",
    ),
    (
        "Почему термос замедляет остывание напитка?",
        "Его стенки уменьшают передачу тепла за счёт теплопроводности, движения воздуха и теплового излучения.",
    ),
    (
        "Для чего на тропах ставят указатели расстояния?",
        "Указатели помогают посетителям оценить путь, время и выбрать маршрут по своим возможностям.",
    ),
    (
        "Почему фотограф хранит негативы отдельно от отпечатков?",
        "Раздельное хранение снижает риск одновременной утраты обоих носителей и позволяет повторно изготовить отпечаток.",
    ),
    (
        "Чем репетиция полезна перед публичным выступлением?",
        "На репетиции участники согласуют темп, переходы и исправляют ошибки до встречи со зрителями.",
    ),
    (
        "Почему архивные коробки делают из бескислотного картона?",
        "Бескислотный материал медленнее разрушает бумагу и помогает документам дольше сохраняться.",
    ),
    (
        "Зачем перед походом проверяют прогноз погоды?",
        "Прогноз помогает подобрать одежду, оценить риски и при необходимости изменить маршрут.",
    ),
    (
        "Почему телескопу дают остыть перед наблюдением?",
        "Когда температура телескопа сравнивается с наружной, воздушные потоки внутри меньше искажают изображение.",
    ),
    (
        "Для чего образцам присваивают каталожные номера?",
        "Уникальный номер связывает предмет с описанием, происхождением и историей хранения.",
    ),
    (
        "Почему свежую краску защищают от пыли?",
        "Пыль прилипает к невысохшему слою, портит поверхность и усложняет последующую отделку.",
    ),
    (
        "Зачем сверять часы перед полевыми наблюдениями?",
        "Единое точное время позволяет сопоставлять записи разных наблюдателей и приборов.",
    ),
)

CLEANUP_CASES = (
    "Полевой журнал хранится в сухом футляре.",
    "После замера наблюдатель подписал карточку.",
    "Новая этикетка закреплена на внутренней стороне коробки.",
    "Маршрут отмечен на бумажной карте синим карандашом.",
    "Резервный фонарь лежит рядом с аптечкой.",
    "Смотритель закрыл витрину после вечернего обхода.",
    "Температуру воды записали сразу после отбора пробы.",
    "Перед началом сеанса проверили аварийное освещение.",
    "Каждый снимок получил дату и краткое описание.",
    "Список участников лежит у дежурного на входе.",
    "После репетиции инструменты вернули в чехлы.",
    "Указатель установили у развилки лесной тропы.",
    "Образец ткани завернули в нейтральную бумагу.",
    "На причале обновили предупреждающую разметку.",
    "Старую плёнку перемотали на чистую катушку.",
    "Перед печатью художник утвердил пробный оттиск.",
    "В журнале связи указали частоту и время сеанса.",
    "Крышку контейнера закрыли после проверки пломбы.",
    "Экскурсовод пересчитал группу перед отправлением.",
    "Данные датчика перенесли в итоговую ведомость.",
)
HAN_INSERTIONS = (
    "系统",
    "路径",
    "完成",
    "错误",
    "数据",
    "服务",
    "文件",
    "请求",
    "消息",
    "响应",
    "状态",
    "检查",
    "记录",
    "时间",
    "结果",
    "成功",
    "更新",
    "测试",
    "设备",
    "安全",
)

CHINESE_CASES = (
    ("为什么观察星星时要避开强光？", "强光会让眼睛难以适应黑暗，也会降低人们看见暗星的能力。"),
    ("地图上的比例尺有什么作用？", "比例尺表示图上距离与实际距离之间的对应关系。"),
    ("为什么纸质档案需要保持干燥？", "潮湿会使纸张变形，并增加霉菌生长和材料损坏的风险。"),
    ("排练为什么能改善演出效果？", "排练让参与者提前协调节奏、衔接和分工，并及时发现问题。"),
    ("为什么采集水样后要立即贴标签？", "立即标记可以避免样品混淆，并保留地点和时间等关键信息。"),
    ("博物馆为什么限制展柜附近的光照？", "较弱的光照可以减缓敏感材料褪色和老化。"),
    ("远足前检查天气预报有什么好处？", "天气预报有助于选择装备、评估风险并调整路线。"),
    ("为什么测量时要记录使用的单位？", "记录单位才能正确解释数值，并与其他测量结果进行比较。"),
    ("备份文件为什么要定期验证？", "定期验证可以确认文件仍能读取，并能在需要时真正恢复。"),
    ("图书为什么不宜紧贴潮湿的墙面？", "墙面的湿气可能进入书页，导致变形、异味或霉变。"),
    ("为什么桥梁需要定期检查连接部位？", "连接部位承受反复载荷，定期检查能及时发现松动、腐蚀或裂纹。"),
    ("录音前为什么要先测试麦克风？", "测试可以确认音量、噪声和连接状态，避免正式录音出现不可补救的问题。"),
    ("种子袋上为什么要写明批次？", "批次信息便于追踪来源、储存条件和后续发芽表现。"),
    ("夜间观测为什么常用红色手电？", "较暗的红光对夜视适应影响较小，同时仍能照亮近处的记录本。"),
    ("陶瓷烧制后为什么要缓慢降温？", "缓慢降温能减少材料内外温差，从而降低开裂风险。"),
    ("为什么展品搬运前要拍摄状态照片？", "照片可以记录搬运前的状况，便于核对运输过程中是否发生变化。"),
    ("野外记录为什么要写准确时间？", "准确时间便于把观察结果与天气、仪器数据和其他记录对应起来。"),
    ("为什么印刷前要制作试印样张？", "试印可以提前检查颜色、位置和文字，避免整批印刷后才发现错误。"),
    ("望远镜为什么需要稳固的支架？", "稳固的支架能减少晃动，使目标更容易保持在视野中。"),
    ("为什么急救箱要定期清点？", "定期清点可以补充缺少的物品，并及时更换过期用品。"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _split(index: int) -> str:
    return "validation" if index < 10 else "test"


def _split_text(index: int, *, validation: str, test: str) -> str:
    return validation if _split(index) == "validation" else test


def _contract(*, language: str = "ru", allow_han: bool = False, blocks: Iterable[str] = ()) -> dict[str, Any]:
    required = list(blocks)
    return {
        "requested_language": language,
        "allow_han": allow_han,
        "allow_han_in_blockquotes": False,
        "require_markdown": bool(required),
        "required_markdown_blocks": required,
    }


def _row(
    family: str,
    index: int,
    prompt: str,
    response: str,
    contract: dict[str, Any],
    tags: Iterable[str],
) -> dict[str, Any]:
    split = _split(index)
    example_id = f"heldout-{family}-{split}-{index % 10:02d}"
    return {
        "id": example_id,
        "split": split,
        "prompt": prompt,
        "response": response,
        "system": DEFAULT_SYSTEM,
        "contract": contract,
        "tags": sorted({"heldout-challenge", family, *tags}),
        "use_for_constraint_rl_smoke": False,
        "review": {
            "status": "accepted",
            "reviewer": REVIEWER,
            "method": "deterministic-template-audit",
            "notes": "project-authored reference checked for language and structural contract",
        },
        "provenance": {
            "dataset": DATASET,
            "revision": DATASET_REVISION,
            "license": "apache-2.0",
            "source_split": "project-authored-heldout",
            "source_record_id": example_id,
        },
    }


def generate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (title, items) in enumerate(LIST_CASES):
        prompt = _split_text(
            index,
            validation=(
                f"Контроль оформления для темы «{title}»: создай раздел Markdown, "
                f"затем расположи по номерам действия {'; '.join(items)}."
            ),
            test=(
                f"Из сведений {'; '.join(items)} подготовь рабочую памятку «{title}». "
                "Ответ должен содержать Markdown-заголовок и упорядоченные пункты."
            ),
        )
        response = f"## {title}\n\n" + "\n".join(
            f"{number}. {item.capitalize()}." for number, item in enumerate(items, start=1)
        )
        rows.append(
            _row(
                "markdown-list",
                index,
                prompt,
                response,
                _contract(blocks=("heading", "list")),
                ("markdown", "russian"),
            )
        )

    for index, (context, *facts) in enumerate(TABLE_CASES):
        fact_text = "; ".join(f"{key} — {value}" for key, value in facts)
        prompt = _split_text(
            index,
            validation=(
                f"Для контрольной темы «{context}» сверстай данные {fact_text} в "
                "Markdown-таблице с колонками «Параметр» и «Значение»."
            ),
            test=(
                f"Сведения наблюдения для темы «{context}» ({fact_text}) нужно "
                "представить двумя табличными столбцами Markdown."
            ),
        )
        response = "| Параметр | Значение |\n| --- | --- |\n" + "\n".join(
            f"| {key} | {value} |" for key, value in facts
        )
        rows.append(
            _row(
                "markdown-table",
                index,
                prompt,
                response,
                _contract(blocks=("table",)),
                ("markdown", "russian"),
            )
        )

    for index, (name, purpose, code) in enumerate(CODE_CASES):
        prompt = _split_text(
            index,
            validation=(
                f"Для проверки кода реализуй на Python функцию `{name}`: она {purpose}. "
                "Перед fenced-блоком кратко объясни решение по-русски."
            ),
            test=(
                f"Задача для функции `{name}` — она {purpose}. Покажи законченную "
                "реализацию внутри блока Markdown и после него дай русское пояснение."
            ),
        )
        response = _split_text(
            index,
            validation=f"Функция {purpose}.\n\n```python\n{code}\n```",
            test=f"```python\n{code}\n```\n\nТакой вариант {purpose}.",
        )
        rows.append(
            _row(
                "markdown-code",
                index,
                prompt,
                response,
                _contract(blocks=("code",)),
                ("code", "markdown", "russian"),
            )
        )

    for index, (title, facts, conclusion, url) in enumerate(MIXED_CASES):
        prompt = _split_text(
            index,
            validation=(
                f"Собери контрольную Markdown-карточку «{title}»: перечисли маркерами "
                f"{'; '.join(facts)}, жирно отметь, что {conclusion}, и добавь {url}."
            ),
            test=(
                f"Оформи сведения {', '.join(facts)} как краткую заметку с заголовком "
                f"«{title}». Нужны список, ссылка {url} и выделенный итог: {conclusion}."
            ),
        )
        response = (
            f"## {title}\n\n"
            + "\n".join(f"- {fact.capitalize()}." for fact in facts)
            + f"\n\n**Вывод:** {conclusion.capitalize()}.\n\n"
            + f"[Подробная карточка]({url})"
        )
        rows.append(
            _row(
                "markdown-mixed",
                index,
                prompt,
                response,
                _contract(blocks=("heading", "list")),
                ("link", "markdown", "russian", "strong"),
            )
        )

    for index, (prompt, response) in enumerate(GENERAL_RUSSIAN_CASES):
        rows.append(
            _row(
                "general-russian",
                index,
                prompt,
                response,
                _contract(),
                ("accidental-han-spontaneous", "russian", "semantic-retention"),
            )
        )

    for index, clean in enumerate(CLEANUP_CASES):
        words = clean.split()
        insertion_position = max(1, len(words) // 2)
        corrupted = " ".join([*words[:insertion_position], HAN_INSERTIONS[index], *words[insertion_position:]])
        prompt = _split_text(
            index,
            validation=(
                "Очисти русское контрольное предложение от одного постороннего "
                f"китайского слова, не меняя смысл: {corrupted}"
            ),
            test=(
                "Найди внедрённую иероглифическую вставку и верни только восстановленную "
                f"русскую фразу без дополнительных комментариев: {corrupted}"
            ),
        )
        rows.append(
            _row(
                "han-cleanup",
                index,
                prompt,
                clean,
                _contract(),
                ("accidental-han", "russian"),
            )
        )

    for index, (prompt, response) in enumerate(CHINESE_CASES):
        rows.append(
            _row(
                "han-retention",
                index,
                prompt,
                response,
                _contract(language="zh", allow_han=True),
                ("chinese-retention", "semantic-retention"),
            )
        )
    return rows


def _family_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [row for row in rows if family in row["tags"]]


def validate_challenge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = validate_rows(rows)
    split_counts = Counter(row["split"] for row in rows)
    if set(split_counts) != {"validation", "test"}:
        raise ValueError("challenge must contain validation and test rows only")
    if any(row["use_for_constraint_rl_smoke"] for row in rows):
        raise ValueError("held-out challenge rows cannot be enabled for training")

    target_prompt_leaks: list[str] = []
    for row in rows:
        normalized_prompt = " ".join(row["prompt"].casefold().split())
        normalized_response = " ".join(row["response"].casefold().split())
        if normalized_response in normalized_prompt:
            target_prompt_leaks.append(row["id"])
    if target_prompt_leaks:
        raise ValueError("reference response appears verbatim in prompt: " + ", ".join(target_prompt_leaks[:8]))

    coverage: dict[str, dict[str, dict[str, int | str]]] = {}
    for split in ("validation", "test"):
        coverage[split] = {}
        for family in EXPECTED_FAMILIES:
            selected = [row for row in _family_rows(rows, family) if row["split"] == split]
            cluster_count = len({evaluation_cluster_id(row) for row in selected})
            status = (
                "PASS" if len(selected) >= MINIMUM_SLICE_ROWS and cluster_count >= MINIMUM_SLICE_CLUSTERS else "FAIL"
            )
            coverage[split][family] = {
                "row_count": len(selected),
                "cluster_count": cluster_count,
                "status": status,
            }
    failed = [
        f"{split}/{family}"
        for split in coverage
        for family, detail in coverage[split].items()
        if detail["status"] != "PASS"
    ]
    if failed:
        raise ValueError("underpowered challenge slices: " + ", ".join(failed))
    return {
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "target_prompt_leak_count": 0,
        "coverage": coverage,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_challenge(clean_view_root: Path, output_dir: Path) -> dict[str, Any]:
    train_path = clean_view_root / "train_rows.jsonl"
    if clean_view_root.name != CLEAN_VIEW_REVISION:
        raise ValueError(f"expected clean view {CLEAN_VIEW_REVISION}")
    if _sha256(train_path) != CLEAN_TRAIN_ROWS_SHA256:
        raise ValueError("clean-v4 train row SHA-256 drift")

    rows = validate_rows(generate_rows())
    validation = validate_challenge(rows)
    clean_train_rows = read_jsonl(train_path)
    # The clean view has already passed source-resolution checks.  This audit
    # independently compares every visible train/eval prompt and response.  We
    # remove only external source hashes so the source resolver does not demand
    # the separately archived Wikipedia source sample; visible text and source
    # record identities remain unchanged.
    audit_train_rows = []
    for row in clean_train_rows:
        audit_row = {**row, "provenance": dict(row["provenance"])}
        audit_row["provenance"].pop("source_text_sha256", None)
        audit_train_rows.append(audit_row)
    combined = [*audit_train_rows, *rows]
    audit = audit_rows(
        combined,
        near_threshold=PRODUCTION_NEAR_DUPLICATE_THRESHOLD,
        inline_source_datasets=INLINE_SOURCE_DATASETS,
    )
    if audit["status"] != "PASS" or any(audit["counts"].values()):
        raise ValueError("held-out challenge overlaps clean-v4 train or another split")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "eval_contracts.jsonl"
    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    audit_path = output_dir / "split_train_contamination_audit.json"
    _write_jsonl(rows_path, rows)
    _write_jsonl(validation_path, [row for row in rows if row["split"] == "validation"])
    _write_jsonl(test_path, [row for row in rows if row["split"] == "test"])
    audit_path.write_text(_canonical_json(audit), encoding="utf-8")

    artifacts = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (rows_path, validation_path, test_path, audit_path)
    }
    manifest = {
        "schema_version": 1,
        "status": "LOCAL-DATA-PASS/MODEL-RESULTS-NOT-RUN",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "scope": "validation-and-test-only-quality-challenge",
        "model_result_count": 0,
        "minimum_slice_rows": MINIMUM_SLICE_ROWS,
        "minimum_slice_clusters": MINIMUM_SLICE_CLUSTERS,
        **validation,
        "clean_train_binding": {
            "dataset_revision": CLEAN_VIEW_REVISION,
            "train_rows": len(clean_train_rows),
            "train_rows_sha256": CLEAN_TRAIN_ROWS_SHA256,
        },
        "contamination_audit": {
            "algorithm": audit["algorithm"],
            "combined_row_count": audit["row_count"],
            "combined_split_counts": audit["split_counts"],
            "status": audit["status"],
            "violation_counts": audit["counts"],
            "train_source_projection": (
                "source_text_sha256 omitted from already-verified clean-v4 train "
                "rows; ids, splits, prompts, responses, and source record identities "
                "are unchanged"
            ),
        },
        "artifacts": artifacts,
        "builder": {
            "file": Path(__file__).name,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "review": {
            "reference_reviewer": REVIEWER,
            "reference_method": "deterministic-template-audit",
            "candidate_semantic_review": "PENDING-BLINDED-HUMAN-REVIEW",
        },
        "usage": {
            "generation": "pass eval_contracts.jsonl and select --split validation or test",
            "training": "forbidden; this directory contains no train split or train parquet",
        },
    }
    (output_dir / "manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_view_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = build_challenge(args.clean_view_root, args.output_dir)
    print(_canonical_json(manifest), end="")


if __name__ == "__main__":
    main()
