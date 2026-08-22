import math
from pathlib import Path

import yaml


# Цей модуль зберігає два типи параметрів:
#
# 1. Параметри окремих камер (роздільність, формат, експозиція тощо).
#    Їх читає stereo_cameras.launch.py і передає вузлам camera_ros.
# 2. Параметри стереообробки (пошук disparity / карти глибини).
#    Вони обчислюються з YAML-файлу калібрування та передаються пакету
#    stereo_image_proc у stereo_proc.launch.py.
#
# Великі літери в іменах QUALITY_CAMERA_PROFILE та CAMERA_PROFILES — це
# прийнята в Python домовленість: так позначають константи, які код не повинен
# змінювати після створення.


# Словник (dict) з базовим, якісним профілем камери.
# Запис 'ключ': значення означає один ROS-параметр та його значення.
QUALITY_CAMERA_PROFILE = {
    # Режим читання сенсора IMX219. Сенсор захоплює повне поле 1640x1232.
    'sensor_mode': '1640:1232',

    # Розмір зображення, яке camera_ros публікує у ROS-топік image_raw.
    'width': 1640,
    'height': 1232,

    # YUYV — некомпресований YUV 4:2:2. Дві сусідні точки мають окрему
    # яскравість Y, але спільні компоненти кольору U та V.
    'format': 'YUYV',

    # Поворот зображення на 180 градусів відповідно до монтажу камер.
    'orientation': 180,

    # Межі тривалості кадру в мікросекундах: [мінімум, максимум].
    # Однакові межі фіксують період кадру. 1 000 000 мкс = 1 с, тобто 1 FPS.
    'FrameDurationLimits': [1000000, 1000000],

    # True та False — булеві значення Python. True вмикає автоматичний
    # підбір експозиції та підсилення сенсора алгоритмом libcamera.
    'AeEnable': True,

    # Базові параметри обробки зображення. Значення 0.0 для Brightness
    # нейтральне; 1.0 для Contrast і Sharpness означає стандартний рівень.
    'Brightness': 0.0,
    'Contrast': 1.0,
    'Sharpness': 1.0,

    # Числовий режим шумозаглушення libcamera. Конкретне трактування числа
    # залежить від версії libcamera та її pipeline handler.
    'NoiseReductionMode': 2,
}

# Профіль для обробки в реальному часі.
# Синтаксис **QUALITY_CAMERA_PROFILE розпаковує всі пари ключ/значення з
# якісного профілю в новий словник. Наступні рядки перевизначають лише три
# параметри. Тому формат, орієнтація, AE та інші налаштування залишаються ті ж.
REALTIME_CAMERA_PROFILE = {
    **QUALITY_CAMERA_PROFILE,

    # Сенсор усе ще працює в режимі 1640x1232, але вихід масштабується до
    # 640x480. Менше пікселів суттєво знижує навантаження стереообробки.
    'width': 640,
    'height': 480,

    # 100 000 мкс = 0.1 с на кадр, тобто цільова частота становить 10 FPS.
    'FrameDurationLimits': [100000, 100000],
}

# Окремий live-профіль для стабільного disparity preview. Він навмисно
# дублює геометрію realtime-профілю, але має власне ім'я, щоб його SGBM
# параметри можна було змінювати незалежно від інших режимів.
LIVE_10FPS_CAMERA_PROFILE = {
    **REALTIME_CAMERA_PROFILE,

    # Варіант швидкого live-preview: ISP масштабує повне поле сенсора до
    # 320x240, а 24 FPS на вході дають запас для цільових 20 FPS disparity.
    'width': 320,
    'height': 240,
    'FrameDurationLimits': [41667, 41667],
}

LIVE_10FPS_STEREO_OVERRIDES = {
    'sgbm_mode': 2,
    'correlation_window_size': 5,
    'min_disparity': 0,
    'disparity_range': 48,
    'speckle_size': 0,
    'speckle_range': 0,
    'disp12_max_diff': 0,
    'uniqueness_ratio': 7.0,
    'P1': 200.0,
    'P2': 800.0,
    'queue_size': 2,
}

# CPU StereoBM baseline for 640x480 realtime. Used as the starting point
# before a custom ROS BM node and Vulkan shader stages.
# stereo_algorithm 0 = OpenCV StereoBM (image_geometry / stereo_image_proc).
REALTIME_BM_STEREO_OVERRIDES = {
    'stereo_algorithm': 0,
    'min_disparity': 0,
    'disparity_range': 96,
    'correlation_window_size': 15,
    'texture_threshold': 10,
    'uniqueness_ratio': 10.0,
    'prefilter_cap': 31,
    'speckle_size': 100,
    'speckle_range': 32,
    'disp12_max_diff': -1,
    'queue_size': 2,
}

# Resource-bounded profile for running SGBM together with NCNN YOLO. The
# geometry is identical to the 320x240 live profile. Twelve camera frames per
# second provide enough headroom for at least 8 disparity frames per second.
PERSON_RANGE_CAMERA_PROFILE = {
    **LIVE_10FPS_CAMERA_PROFILE,
    'FrameDurationLimits': [83333, 83333],
}

# Test profile: same 12 FPS budget as person_range, but 640x480 (2x linear
# resolution) instead of 320x240. Uses the already-calibrated native
# 640x480 stereo YAML directly (LIVE_CALIBRATIONS) -- no rescaling needed.
# Expect substantially higher SGBM cost: not just 4x the pixels, but a wider
# disparity_range too, since the focal length in pixels doubles with
# resolution (see PERSON_RANGE_2X_STEREO_OVERRIDES below).
PERSON_RANGE_2X_CAMERA_PROFILE = {
    **REALTIME_CAMERA_PROFILE,
    'FrameDurationLimits': [83333, 83333],
}

# Realistic follow-me distance window: 1.5-5.0m (matches fusion_node's
# actual min_depth/max_depth=0.5..5.0, tightened at the near end -- a
# tracked person realistically isn't closer than ~1.5m to a drone/robot
# camera in this scenario). disparity_range is dominated by the near limit
# (small Z -> large disparity), so this cuts SGBM cost roughly 3x versus
# a literal 0.5m near limit (144) at the native 640x480 f*B=62.415, while
# still covering the far end fusion actually uses (previously capped at an
# arbitrary 3m, under-covering fusion's real 5m ceiling).
PERSON_RANGE_2X_STEREO_OVERRIDES = {
    'sgbm_mode': 2,
    'correlation_window_size': 5,
    'min_disparity': 0,
    'disparity_range': 48,
    'speckle_size': 0,
    'speckle_range': 0,
    'disp12_max_diff': 0,
    'uniqueness_ratio': 7.0,
    'P1': 200.0,
    'P2': 800.0,
    'queue_size': 2,
}

# Теоретично максимальний профіль камер: повне зчитування сенсора без кропу,
# максимальна доступна бітність і цільова частота 30 кадрів на секунду.
THEORETICAL_MAXIMUM_CAMERA_PROFILE = {
    **QUALITY_CAMERA_PROFILE,

    # Повна активна область сенсора IMX219 забезпечує максимальний кут огляду.
    'sensor_mode': '3280:2464',
    'width': 3280,
    'height': 2464,

    # 10-бітний Bayer RGGB у packed CSI-2 форматі зберігає максимум даних
    # сенсора без перетворення в YUV.
    'format': 'SRGGB10_CSI2P',

    # 33 333 мкс на кадр відповідають приблизно 30 FPS.
    'FrameDurationLimits': [33333, 33333],
}

# Повнорозмірний профіль для практичного тесту SGBM. На відміну від
# ``theoretical_maximum``, вихід проходить через ISP у формат YUYV, який
# image_proc може перетворити на mono8. Частоту навмисно обмежено до 1 FPS:
# один повний кадр 3280x2464 і так є важким навантаженням для CPU SGBM.
MAXIMUM_SGBM_CAMERA_PROFILE = {
    **QUALITY_CAMERA_PROFILE,
    'sensor_mode': '3280:2464',
    'width': 3280,
    'height': 2464,
    'format': 'YUYV',
    'FrameDurationLimits': [1000000, 1000000],
}

# Таблиця доступних профілів. Рядок, переданий launch-аргументом profile,
# використовується як ключ: CAMERA_PROFILES['quality'], ['realtime'] або
# ['theoretical_maximum'].
CAMERA_PROFILES = {
    'quality': QUALITY_CAMERA_PROFILE,
    'realtime': REALTIME_CAMERA_PROFILE,
    'live_10fps': LIVE_10FPS_CAMERA_PROFILE,
    # Foxglove-профіль використовує той самий перевірений режим камер і
    # SGBM, але launch-файл не запускає локальні RViz/disparity_viz.
    'foxglove_live': LIVE_10FPS_CAMERA_PROFILE,
    'person_range': PERSON_RANGE_CAMERA_PROFILE,
    'person_range_2x': PERSON_RANGE_2X_CAMERA_PROFILE,
    'theoretical_maximum': THEORETICAL_MAXIMUM_CAMERA_PROFILE,
    'maximum_sgbm': MAXIMUM_SGBM_CAMERA_PROFILE,
}

# Старе ім'я залишене для сумісності з кодом або документацією, які можуть
# імпортувати CAMERA_PROFILE. Воно посилається на якісний профіль.
CAMERA_PROFILE = QUALITY_CAMERA_PROFILE


def get_camera_profile(name: str) -> dict[str, object]:
    """Повернути незалежну копію профілю камери за його назвою.

    ``name: str`` — підказка типу: функція очікує рядок.
    ``-> dict[str, object]`` — функція повертає словник із рядковими ключами;
    його значення можуть мати різні типи (int, float, bool, str або list).
    """
    try:
        # Доступ до словника за ключем. Для невідомого ключа Python створить
        # виняток KeyError, який обробляється нижче.
        profile = CAMERA_PROFILES[name]
    except KeyError as exc:
        # sorted(...) впорядковує назви, а ', '.join(...) об'єднує їх у
        # зручний для повідомлення рядок: "quality, realtime".
        choices = ', '.join(sorted(CAMERA_PROFILES))
        raise ValueError(
            f'unknown camera profile {name!r}; expected one of: {choices}'
        ) from exc

    # dict(profile) створює поверхневу копію. Launch-файл додає до отриманого
    # словника camera та frame_id, тому не можна повертати глобальний словник
    # напряму — інакше спільна константа могла б випадково змінитися.
    result = dict(profile)

    # FrameDurationLimits є вкладеним змінюваним списком. Поверхнева копія
    # словника не копіює вкладені об'єкти, тому список копіюється окремо.
    result['FrameDurationLimits'] = list(profile['FrameDurationLimits'])
    return result


def calculate_disparity_search(
    f_times_baseline: float,
    near_m: float = 0.5,
    far_m: float = 3.0,
) -> tuple[int, int]:
    """Обчислити діапазон disparity для заданого діапазону глибин.

    Для ректифікованої стереопари використовується формула ``d = fB / Z``:
    d — disparity у пікселях, f — фокусна відстань у пікселях,
    B — база між камерами в метрах, Z — відстань до об'єкта в метрах.

    Повертається кортеж ``(min_disparity, disparity_range)``. Кортеж — це
    незмінна впорядкована пара значень Python.
    """
    # Перевіряємо фізично коректні вхідні дані: fB і відстані додатні,
    # а дальня межа справді розташована далі за ближню.
    if f_times_baseline <= 0 or near_m <= 0 or far_m <= near_m:
        raise ValueError('expected fB > 0 and 0 < near_m < far_m')

    # Далекі об'єкти мають менший disparity, близькі — більший.
    far_disparity = f_times_baseline / far_m
    near_disparity = f_times_baseline / near_m

    # Нижню межу опускаємо ще на 8 пікселів як запас і округлюємо вниз до
    # кратного 8. max(0, ...) не дозволяє отримати від'ємну межу.
    minimum = max(0, math.floor((far_disparity - 8.0) / 8.0) * 8)

    # Верхній край збільшуємо на 5% і округлюємо вгору, щоб діапазон точно
    # охоплював бажану ближню відстань навіть при похибках калібрування.
    required_maximum = math.ceil(near_disparity * 1.05)

    # OpenCV StereoBM/SGBM вимагає, щоб кількість disparity була кратна 16.
    # ``+ 1`` враховує обидві включені межі діапазону.
    disparity_range = math.ceil(
        (required_maximum - minimum + 1) / 16.0
    ) * 16
    return minimum, disparity_range


def load_stereo_profile(
    calibration_path: Path,
    expected_size: tuple[int, int] = (1640, 1232),
) -> dict[str, object]:
    """Створити параметри stereo_image_proc з YAML-калібрування.

    ``calibration_path`` — шлях pathlib.Path до калібрування правої камери.
    ``expected_size`` — очікувана пара (ширина, висота); значення після ``=``
    є типовим і використовується, якщо виклик не передав інший розмір.
    """
    # read_text() читає весь файл як текст, а safe_load() перетворює YAML на
    # стандартні об'єкти Python: словники, списки, числа та рядки.
    data = yaml.safe_load(calibration_path.read_text())

    # int(...) гарантує цілочисельний тип навіть тоді, коли YAML-парсер
    # повернув сумісне числове представлення.
    width = int(data['image_width'])
    height = int(data['image_height'])

    # Калібрування дійсне лише для роздільності, на якій воно було отримане.
    # Порівняння двох кортежів одразу перевіряє і ширину, і висоту.
    if (width, height) != expected_size:
        # Розпакування кортежу у дві окремі змінні.
        expected_width, expected_height = expected_size
        raise ValueError(
            f'expected {expected_width}x{expected_height} calibration, '
            f'got {width}x{height}'
        )

    # Проєкційна матриця P має розмір 3x4, отже містить рівно 12 чисел.
    projection = data['projection_matrix']['data']
    if len(projection) != 12:
        raise ValueError('projection_matrix.data must contain 12 values')

    # Для правої ректифікованої камери елемент P[0,3] (індекс 3 у плоскому
    # списку) дорівнює приблизно -fx * B. abs(...) прибирає знак, а float(...)
    # гарантує тип з плаваючою крапкою.
    f_times_baseline = abs(float(projection[3]))

    # Функція повертає пару, яку тут одразу розпаковуємо у дві змінні.
    minimum, disparity_range = calculate_disparity_search(f_times_baseline)

    # Повернений словник напряму стає набором параметрів stereo_image_proc.
    return {
        # 1 обирає алгоритм Semi-Global Block Matching (SGBM).
        'stereo_algorithm': 1,

        # Режим OpenCV SGBM з числовим значенням 2 (SGBM 3-way).
        'sgbm_mode': 2,

        # Межа попереднього обрізання інтенсивності пікселів.
        'prefilter_cap': 31,

        # Розмір квадратного вікна зіставлення, тут 7x7 пікселів.
        'correlation_window_size': 7,

        # Нижня межа і загальна кількість перевірюваних disparity.
        'min_disparity': minimum,
        'disparity_range': disparity_range,

        # 0 вимикає відсіювання областей за локальною текстурою.
        'texture_threshold': 0,

        # Видалення малих ізольованих областей disparity (speckles):
        # максимальний розмір області та допустима різниця disparity в ній.
        'speckle_size': 150,
        'speckle_range': 2,

        # Максимально допустима різниця при перевірці left-right consistency.
        'disp12_max_diff': 1,

        # Наскільки найкраще зіставлення має переважати наступного кандидата.
        'uniqueness_ratio': 7.0,

        # Штрафи SGBM за зміну disparity між сусідніми пікселями.
        # Для одноканального зображення та вікна 7: P1=8*1*7^2,
        # P2=32*1*7^2. P2 > P1 сильніше карає великі стрибки disparity.
        'P1': 392.0,
        'P2': 1568.0,

        # False створює монохромну, а не кольорову хмару точок.
        'use_color': False,

        # Не додавати вирівнювальні байти (padding) у PointCloud2, щоб
        # зменшити обсяг даних і спростити подальше читання хмари.
        'avoid_point_cloud_padding': True,
    }


def load_named_stereo_profile(
    name: str,
    calibration_path: Path,
    expected_size: tuple[int, int],
) -> dict[str, object]:
    """Load geometry-derived parameters and apply named mode overrides."""
    profile = load_stereo_profile(calibration_path, expected_size)
    if name in ('live_10fps', 'foxglove_live', 'person_range'):
        profile.update(LIVE_10FPS_STEREO_OVERRIDES)
    elif name == 'person_range_2x':
        profile.update(PERSON_RANGE_2X_STEREO_OVERRIDES)
    elif name in ('realtime_bm', 'cpu_stereobm_baseline'):
        profile.update(REALTIME_BM_STEREO_OVERRIDES)
    return profile
