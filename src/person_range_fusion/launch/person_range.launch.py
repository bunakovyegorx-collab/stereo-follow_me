"""Resource-bounded stereo + YOLO + person range with Foxglove host offload.

CPU placement (Pi 5, 4 cores):
  cores 0-2: cameras / SGBM container / depth convert / diagnostics
  core 3:    YOLO + fusion (rate-limited)

Full pipeline always includes foxglove_bridge (:8765), rqt_image_view,
rqt_graph, and a reliable image_view on the Pi.
Heavy pixel debug_image stays off by default; host uses annotations + depth + scene.
"""

# ============================================================================
# ЦЕ ЗВИЧАЙНИЙ PYTHON-ФАЙЛ.
# У ROS 2 launch-файл — це не конфіг (як .yaml чи .xml), а python-скрипт,
# який ПОВИНЕН містити функцію generate_launch_description().
# Команда `ros2 launch <пакет> person_range.launch.py` імпортує цей файл
# і викликає саме цю функцію. Все, що вона поверне (об'єкт LaunchDescription),
# і буде запущено.
# ============================================================================

# --- ІМПОРТИ -----------------------------------------------------------
# launch / launch_ros — це два python-пакети ROS 2, які дають "будівельні
# блоки" для опису того, що і як запускати. Нижче — що саме ми беремо
# з кожного і навіщо.

from launch import LaunchDescription
# LaunchDescription — контейнер верхнього рівня. Функція generate_launch_description()
# має повернути САМЕ ОБ'ЄКТ цього класу — список усіх дій (нод, аргументів,
# інших launch-файлів), які треба виконати.

from launch.actions import DeclareLaunchArgument
# DeclareLaunchArgument — оголошує "параметр запуску" (launch argument).
# Це аргументи, які можна передати з командного рядка, наприклад:
#   ros2 launch person_range_fusion person_range.launch.py visualization:=true
# Кожен такий аргумент має ім'я та значення за замовчуванням (default_value).

from launch.actions import IncludeLaunchDescription
# IncludeLaunchDescription — дозволяє "вставити" ЦІЛИЙ ІНШИЙ launch-файл
# всередину цього, ніби copy-paste. Так ми підключаємо launch-файл
# з пакета drone_stereo_bringup (запуск камер), не дублюючи його код тут.

from launch.actions import TimerAction
# TimerAction — "зроби це, але через N секунд після старту launch".
# Використовується нижче для GUI-вікон (rqt_image_view тощо), яким треба
# почекати, поки топіки з даними вже з'являться, інакше вікно відкриється
# "в порожнечу".

from launch.conditions import IfCondition
# IfCondition — умовний запуск ноди: "запускай ЦЮ ноду, ТІЛЬКИ ЯКЩО
# значення такого-то launch-аргументу == true". Це аналог `if` для нод.

from launch.launch_description_sources import PythonLaunchDescriptionSource
# PythonLaunchDescriptionSource — каже IncludeLaunchDescription: "той файл,
# який я підключаю, написаний на Python" (а не .xml/.yaml launch-формат,
# які теж існують у ROS 2, але тут не використовуються).

from launch.substitutions import EnvironmentVariable
# EnvironmentVariable('HOME') — "підстав сюди значення змінної середовища
# $HOME операційної системи". Обчислюється не зараз (коли Python читає файл),
# а вже під час запуску launch. Такі речі звуться "substitutions" (підстановки)
# — вони як плейсхолдери, що заповнюються пізніше.

from launch.substitutions import LaunchConfiguration
# LaunchConfiguration('foxglove') — "підстав сюди ЗНАЧЕННЯ launch-аргументу
# з іменем foxglove" (той самий, що оголошений через DeclareLaunchArgument
# вище). Це те, як різні частини launch-файлу читають аргументи один одного.

from launch.substitutions import PathJoinSubstitution
# PathJoinSubstitution(['a', 'b', 'c']) — те саме, що os.path.join('a','b','c'),
# але працює з підстановками (не з простими рядками), тому це не можна
# замінити на f-string чи звичайний +.

from launch_ros.actions import Node
# Node — головний "будівельний блок": опис ОДНІЄЇ ROS 2-ноди для запуску
# (з якого пакета, який виконуваний файл, з якими параметрами).
# Це те, що реально стає окремим процесом в ОС.

from launch_ros.substitutions import FindPackageShare
# FindPackageShare('drone_stereo_bringup') — "знайди, де на диску встановлений
# (в install/) пакет з таким іменем, і поверни шлях до його теки share/".
# Використовується нижче, щоб знайти launch-файл ІНШОГО пакета за іменем
# пакета, а не за жорстко прописаним абсолютним шляхом.


# ============================================================================
# СПИСКИ ТОПІКІВ ДЛЯ FOXGLOVE_BRIDGE
# Це звичайні python-списки рядків (list[str]). Кожен рядок — це
# РЕГУЛЯРНИЙ ВИРАЗ (regex), який foxglove_bridge звіряє з іменами топіків.
# Якщо топік підпадає під якийсь патерн зі списку — його видно у Foxglove
# Studio, якщо ні — приховано. Це зроблено, щоб не заливати слабкий Wi-Fi
# на Pi усіма топіками підряд, а показувати тільки потрібні для дебагу.
# ============================================================================

# Перший список: топіки, які взагалі дозволено віддавати в Foxglove.
# ^ і $ у regex означають "рівно цей рядок повністю", без нічого зайвого
# до чи після (інакше '/stereo/depth' підхопив би і '/stereo/depth/extra').
PERSON_RANGE_FOXGLOVE_TOPIC_WHITELIST = [
    '^/stereo/disparity$',
    '^/stereo/depth$',
    '^/stereo/left/camera_info$',
    '^/stereo/left/image_rect$',
    '^/person_range/image_annotations$',
    '^/person_range/scene$',
    '^/person_range/nearest_point$',
    '^/person_range/detections_3d$',
    '^/person_range/diagnostics$',
    '^/person_detector/detections$',
    '^/tf$',
    '^/tf_static$',
    '^/rosout$',
    '^/foxglove_bridge/client_count$',
]

# Другий список — підмножина першого: топіки, для яких дозволено ще й
# "best effort" QoS (гарантована доставка не обов'язкова — швидше,
# але повідомлення можуть губитись; підходить для відео/картинок,
# де важливіша свіжість кадру, ніж 100% доставка кожного кадру).
PERSON_RANGE_FOXGLOVE_BEST_EFFORT = [
    '^/stereo/disparity$',
    '^/stereo/depth$',
    '^/stereo/left/image_rect$',
    '^/person_range/image_annotations$',
    '^/person_range/scene$',
    '^/person_range/nearest_point$',
    '^/person_range/detections_3d$',
    '^/person_range/diagnostics$',
    '^/person_detector/detections$',
]


# ============================================================================
# ГОЛОВНА ФУНКЦІЯ — саме її шукає та викликає `ros2 launch`.
# Синтаксис `-> LaunchDescription` — це просто підказка типу (type hint):
# "ця функція повертає об'єкт LaunchDescription". На виконання не впливає,
# лише для читабельності та статичних перевірок типів (mypy тощо).
# ============================================================================
def generate_launch_description() -> LaunchDescription:

    # --- Значення за замовчуванням для шляхів (обчислюються "лениво") -----
    # PathJoinSubstitution НЕ виконує os.path.join прямо зараз. Він створює
    # об'єкт-"рецепт", який ros2 launch виконає пізніше, підставивши
    # реальне значення $HOME. Тому це виглядає як список рядків, а не
    # як звичайний Python-рядок з /.
    model_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.local', 'share', 'camera-yolo', 'models',
        'yolo26n_ncnn_model',
    ])
    # Результат еквівалентний: $HOME/.local/share/camera-yolo/models/yolo26n_ncnn_model
    # Тобто шлях до директорії з YOLO-моделлю.

    python_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.venvs', 'camera-yolo', 'bin', 'python',
    ])
    # Результат: $HOME/.venvs/camera-yolo/bin/python
    # Це шлях до python-інтерпретатора з ОКРЕМОГО virtualenv, спеціально
    # створеного для YOLO (бо YOLO має свої залежності, які не хочемо
    # мішати з системним ROS-python).

    # --- "Локальні змінні"-скорочення для читання launch-аргументів -------
    # Кожен рядок нижче — це НЕ значення (не True/False), а "посилання":
    # "коли прийде час — прочитай, яке значення має launch-аргумент з таким
    # іменем". Робимо це заздалегідь, щоб нижче писати коротше (visualization
    # замість LaunchConfiguration('visualization') щоразу).
    visualization = LaunchConfiguration('visualization')
    foxglove = LaunchConfiguration('foxglove')
    rqt_image_view = LaunchConfiguration('rqt_image_view')
    rqt_graph = LaunchConfiguration('rqt_graph')
    image_view = LaunchConfiguration('image_view')

    # --- return LaunchDescription([ ... ]) ---------------------------------
    # Усе, що передається в LaunchDescription — це ОДИН СПИСОК (Python list)
    # з "діями" (actions). Порядок елементів у списку — це приблизний
    # порядок запуску (але майже все стартує практично одночасно, окрім
    # того, що явно загорнуте в TimerAction).
    return LaunchDescription([

        # ====================================================================
        # БЛОК 1: ОГОЛОШЕННЯ ПАРАМЕТРІВ ЗАПУСКУ (DeclareLaunchArgument)
        # Кожен такий рядок — це один параметр, який можна перевизначити
        # з командного рядка через синтаксис ім'я:=значення, наприклад:
        #   ros2 launch person_range_fusion person_range.launch.py yolo_fps:=5.0
        # Якщо нічого не передати — використовується default_value.
        # ====================================================================

        DeclareLaunchArgument('model_path', default_value=model_default),
        # Шлях до YOLO-моделі. За замовчуванням — model_default, обчислений вище.

        DeclareLaunchArgument('python_executable', default_value=python_default),
        # Який python-інтерпретатор запускати для YOLO-ноди (окремий venv).

        DeclareLaunchArgument('confidence_threshold', default_value='0.40'),
        # Поріг впевненості YOLO: детекції з confidence нижче 0.40 ігноруються.
        # Зверніть увагу: значення передається як РЯДОК '0.40', а не число —
        # так влаштований launch-аргумент, він завжди рядок; у python-коді
        # ноди його вже сконвертують у float.

        DeclareLaunchArgument('imgsz', default_value='320'),
        # Розмір зображення (у пікселях), до якого масштабується кадр
        # перед подачею в YOLO. Менше значення = швидше, але гірша точність.

        DeclareLaunchArgument('yolo_fps', default_value='3.0'),
        # Максимальна частота запуску YOLO-інференсу (кадрів за секунду).
        # Обмежено навмисно — YOLO дорога по CPU, ядро #3 і так навантажене.

        DeclareLaunchArgument(
            'visualization',
            default_value='False',
            description=(
                'Legacy local debug_image + image_view on the Pi. '
                'Prefer rqt_image_view / image_view on image_rect for normal use.'
            ),
        ),
        # visualization=True вмикає СТАРИЙ важкий режим налагодження
        # (малювання debug-картинки з рамками прямо на Pi). За замовчуванням
        # вимкнено, бо це зайве навантаження — краще дивитись через Foxglove.
        # `description=` — це просто текст для довідки (`ros2 launch <file> --show-args`),
        # на роботу не впливає.

        DeclareLaunchArgument(
            'foxglove',
            default_value='True',
            description=(
                'Mandatory for full pipeline: foxglove_bridge on :8765 '
                'with person-range topic whitelist.'
            ),
        ),
        # Чи запускати foxglove_bridge (веб-сокет сервер на порту 8765,
        # через нього Foxglove Studio на телефоні/ноуті бачить дані).

        DeclareLaunchArgument(
            'rqt_image_view',
            default_value='True',
            description=(
                'Mandatory for full pipeline: local rqt_image_view on the Pi.'
            ),
        ),
        # Чи відкривати GUI-вікно rqt_image_view прямо на екрані Raspberry Pi.

        DeclareLaunchArgument(
            'rqt_graph',
            default_value='True',
            description=(
                'Mandatory for full pipeline: rqt_graph on the Pi display.'
            ),
        ),
        # Чи відкривати rqt_graph — вікно, яке малює граф "хто з ким
        # спілкується" (ноди + топіки), корисно для дебагу пайплайна.

        DeclareLaunchArgument(
            'image_view',
            default_value='True',
            description=(
                'Mandatory local OpenCV image_view on image_view_topic '
                '(works when rqt C++ plugins are broken).'
            ),
        ),
        # Простіший (запасний) переглядач картинки на базі OpenCV.
        # Тримається як резерв, бо rqt_image_view іноді ламається
        # через C++/SIP-плагіни.

        DeclareLaunchArgument(
            'image_view_topic',
            default_value='/stereo/left/image_rect',
            description='Image topic opened by rqt_image_view / image_view.',
        ),
        # Який саме топік з картинкою відкривати у вьюверах вище.

        DeclareLaunchArgument(
            'display',
            default_value=':0',
            description='X11 DISPLAY for rqt_* and image_view GUIs.',
        ),
        # На якому X11-дисплеї малювати GUI-вікна (":0" — стандартний
        # локальний дисплей монітора, підключеного до Pi).

        # ====================================================================
        # БЛОК 2: ПІДКЛЮЧЕННЯ ІНШОГО LAUNCH-ФАЙЛУ (камери + стерео)
        # ====================================================================
        IncludeLaunchDescription(
            # Джерело — Python launch-файл, шлях до якого будується так:
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('drone_stereo_bringup'),  # "де встановлений пакет drone_stereo_bringup"
                    'launch',                                   # + підтека launch/
                    'stereo_sgbm_light.launch.py',               # + сам файл
                ])
                # Разом: install/drone_stereo_bringup/share/drone_stereo_bringup/launch/stereo_sgbm_light.launch.py
            ),
            # launch_arguments — це аргументи, які МИ передаємо у ВКЛЮЧЕНИЙ
            # launch-файл (аналогічно до ros2 launch ... arg:=value, тільки
            # програмно). .items() перетворює звичайний python dict
            # {'profile': 'person_range', ...} на список пар (ключ, значення),
            # бо такий формат вимагає launch_arguments.
            launch_arguments={
                'profile': 'person_range',       # який профіль розрізнення/налаштувань SGBM використати
                'rviz': 'False',                 # не відкривати RViz звідси (у нас свої вьювери нижче)
                'disparity_viz': 'False',         # не малювати окрему кольорову картинку disparity
                'combined_mode': 'True',          # прапорець "запускається як частина великого пайплайна"
            }.items(),
        ),
        # Цей include фактично "вставляє" сюди запуск: двох camera_ros нод
        # (ліва+права камера), ректифікації та SGBM stereo_image_proc —
        # усе це визначено В ІНШОМУ файлі, ми його просто підключаємо.

        # ====================================================================
        # БЛОК 3: ОКРЕМІ НОДИ ЦЬОГО ПАЙПЛАЙНА (Node(...))
        # Кожен Node(...) — це ОДИН окремий процес в операційній системі.
        # Спільні для всіх параметри Node:
        #   package    — з якого ROS-пакета брати виконуваний файл
        #   executable — ім'я entry point'а (див. setup.py пакета)
        #   name       — як нода буде називатись у графі ROS (ros2 node list)
        #   namespace  — префікс для всіх топіків/сервісів цієї ноди
        #                (наприклад namespace='stereo' + топік 'depth' → /stereo/depth)
        #   parameters — список dict'ів з ROS-параметрами ноди (ros2 param ...)
        #   prefix     — команда, якою "обгортається" запуск виконуваного файлу
        #                (тут використовується для керування CPU/пріоритетом)
        #   output     — куди виводити stdout/stderr ноди ('screen' = у ваш термінал)
        # ====================================================================

        # Phase 3: metric depth next to SGBM on cores 0-2 (not core 3).
        Node(
            package='person_range_fusion',
            executable='depth',      # entry point 'depth' → person_range_fusion/depth_node.py:main
            name='depth',
            namespace='stereo',      # усі топіки цієї ноди матимуть префікс /stereo/...
            prefix=['nice -n 0 taskset -c 0-2'],
            # prefix — це команда ОС, якою запускається сам виконуваний файл, тобто
            # реальна команда буде виглядати як:
            #   nice -n 0 taskset -c 0-2 <шлях_до_executable_depth>
            # nice -n 0     — пріоритет процесу для планувальника ОС (0 = звичайний)
            # taskset -c 0-2 — "прив'яжи цей процес тільки до CPU-ядер 0,1,2"
            #                  (щоб не заважав YOLO/fusion, які сидять на ядрі 3)
            parameters=[{
                'disparity_topic': '/stereo/disparity',
                # Абсолютний шлях '/stereo/disparity' (починається зі '/') —
                # означає "не додавай namespace ноди спереду, топік вже повний".
                'depth_topic': '/stereo/depth',
                # Коментар в оригінальному коді пояснює: якщо писати без
                # ведучого '/', namespace 'stereo' додасться автоматично,
                # тут же він прописаний вручну явно, щоб результат завжди
                # був саме /stereo/depth незалежно від namespace.
            }],
            output='screen',
        ),

        Node(
            package='yolo_person_car',
            executable='detector',   # entry point 'detector' → yolo_person_car/detector_node.py:main
            name='detector',
            namespace='person_detector',
            prefix=[
                'nice -n 10 taskset -c 3 ',
                # nice -n 10 — НИЖЧИЙ пріоритет (число більше = менш пріоритетний
                # для ОС), taskset -c 3 — прив'язка тільки до ядра #3.
                LaunchConfiguration('python_executable'),
                # ВАЖЛИВО: тут в prefix підставляється ще й ШЛЯХ ДО ПИТОН-ІНТЕРПРЕТАТОРА
                # (той самий python_default з окремого venv). Тобто реальна команда:
                #   nice -n 10 taskset -c 3  <venv>/bin/python <шлях_до_executable_detector>
                # Це потрібно, щоб detector_node.py виконувався ІНШИМ python'ом,
                # ніж системний ROS-python (бо в venv стоять залежності YOLO,
                # яких немає/не повинно бути в системному python).
            ],
            additional_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            # additional_env — додаткові змінні середовища ТІЛЬКИ для цього
            # процесу. Тут обмежуємо кількість потоків, які використовують
            # бібліотеки лінійної алгебри (OpenMP/OpenBLAS), інакше вони самі
            # спробують розлізтися по всіх ядрах, ігноруючи taskset.
            parameters=[{
                'image_topic': '/stereo/left/image_rect',   # звідки брати кадри
                'model_path': LaunchConfiguration('model_path'),           # підставиться launch-аргумент
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'imgsz': LaunchConfiguration('imgsz'),
                'allowed_classes': [0],
                # У YOLO кожен клас об'єкта має номер (з датасету COCO).
                # 0 = "person" (людина). Тобто детектор шукає ТІЛЬКИ людей,
                # ігноруючи всі інші класи (машини, тварин тощо).
                'enable_debug_image': False,
                'max_inference_fps': LaunchConfiguration('yolo_fps'),
            }],
            output='screen',
        ),

        # Optical camera frame → REP-103 (X forward, Y left, Z up).
        # Projection / depth / camera_info stay optical; fusion outputs and
        # Foxglove person cubes use stereo_left_up. Quaternion is the standard
        # camera_link ← camera_optical (rpy -90°, 0, -90°).
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            # Це ГОТОВА нода зі стандартного пакета ROS (tf2_ros), не з нашого
            # коду. Вона публікує ОДНЕ незмінне (статичне) перетворення
            # координат між двома системами відліку ("фреймами").
            name='stereo_left_up_tf',
            arguments=[
                # arguments (на відміну від parameters!) — це аргументи
                # командного рядка, як якби ви вручну ввели в терміналі:
                #   ros2 run tf2_ros static_transform_publisher --x 0 --y 0 ...
                '--x', '0', '--y', '0', '--z', '0',
                # зсув (translation) 0,0,0 — фрейми в одній точці простору
                '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                # обертання (rotation) у вигляді кватерніона [x,y,z,w] —
                # переводить систему координат оптики камери в "звичну"
                # систему X-вперед/Y-вліво/Z-вгору (стандарт ROS, REP-103)
                '--frame-id', 'stereo_left_up',
                '--child-frame-id', 'stereo_left_optical_frame',
                # frame-id/child-frame-id — імена батьківського й дочірнього
                # фрейму. Тобто ми кажемо: "stereo_left_optical_frame є
                # дочірнім по відношенню до stereo_left_up, ось таке між
                # ними перетворення". Ці фрейми потім бачить RViz/Foxglove,
                # щоб правильно розмістити 3D-об'єкти в просторі.
            ],
            output='screen',
        ),

        # Phase 1+5: annotations + scene always; legacy debug only if visualization.
        # person_marker_size is natural REP-103: [width, depth, height].
        Node(
            package='person_range_fusion',
            executable='fusion',    # entry point 'fusion' → person_range_fusion/fusion_node.py:main
            name='fusion',
            namespace='person_range',
            prefix=['nice -n 5 taskset -c 3'],
            # Проміжний пріоритет (5) між depth (0) і detector (10),
            # теж прив'язана до ядра #3, разом з YOLO — бо fusion залежить
            # від виводу YOLO і має рахуватись одразу після нього.
            parameters=[{
                'enable_debug_image': visualization,
                # Значення тут — це ЗМІННА visualization, оголошена вище
                # (LaunchConfiguration('visualization')), тобто debug-картинка
                # ноди fusion вмикається/вимикається тим самим аргументом
                # visualization:=true/false з командного рядка.
                'enable_image_annotations': True,   # завжди малювати "легкі" анотації (рамки/підписи) — не картинку, а векторні дані
                'enable_scene': True,                 # завжди публікувати 3D-сцену (для Foxglove 3D панелі)
                'output_frame': 'stereo_left_up',     # у якому фреймі виражати координати результату
                'person_marker_size': [0.5, 0.5, 1.7],
                # Розмір "коробки" людини для 3D-візуалізації:
                # [ширина, глибина, висота] в метрах — типовий силует людини.
                'person_marker_wireframe': True,       # малювати як каркас (дротяну рамку), а не суцільну фігуру
                'person_marker_edge_thickness': 0.03,  # товщина ліній каркаса, у метрах
                'person_marker_fill_alpha': 0.0,       # прозорість заливки: 0.0 = повністю прозора (тільки лінії видно)
            }],
            output='screen',
        ),

        # Phase 2: diagnostics off core 3, high priority so it survives load.
        Node(
            package='person_range_fusion',
            executable='pi_diagnostics',
            name='pi_diagnostics',
            namespace='person_range',
            prefix=['nice -n -5 taskset -c 0-2'],
            # nice -n -5 — ВИЩИЙ за звичайний пріоритет (від'ємне число
            # у nice = ВАЖЛИВІШИЙ процес для планувальника). Логіка:
            # діагностика має продовжувати працювати, навіть якщо ядра 0-2
            # перевантажені — щоб ви бачили, що щось пішло не так.
            output='screen',
        ),

        # Mandatory host offload: foxglove_bridge (whitelist, no debug_image).
        # address 0.0.0.0 = listen on ALL interfaces (hotspot, eth0, …).
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            condition=IfCondition(foxglove),
            # condition=IfCondition(foxglove) — ЦЯ НОДА ЗАПУСКАЄТЬСЯ, ТІЛЬКИ
            # ЯКЩО launch-аргумент 'foxglove' == 'True' (за замовчуванням
            # так і є). Якщо запустити з foxglove:=false — ноду взагалі
            # не буде створено.
            parameters=[{
                'address': '0.0.0.0',
                # 0.0.0.0 означає "слухай на ВСІХ мережевих інтерфейсах"
                # (Wi-Fi, Ethernet, hotspot) — тому телефон у тій самій
                # мережі теж може підключитись, не тільки localhost.
                'port': 8765,                    # порт веб-сокет сервера — саме на нього заходить Foxglove Studio
                'min_qos_depth': 1,               # мінімальний розмір черги повідомлень (QoS depth)
                'max_qos_depth': 1,               # максимальний — тобто тримаємо лише 1 останнє повідомлення (не накопичуємо старі)
                'num_threads': 2,                 # скільки потоків обробки на сервері
                'send_buffer_limit': 1000000,     # ліміт буфера відправки в байтах (~1 МБ), захист від переповнення при повільній мережі
                'use_compression': False,          # не стискати дані (стиснення коштує CPU, якого й так мало)
                'best_effort_qos_topic_whitelist': (
                    PERSON_RANGE_FOXGLOVE_BEST_EFFORT
                ),
                # список топіків (той, що вище у файлі), для яких дозволено
                # "best effort" QoS — швидка, але не гарантована доставка
                'topic_whitelist': PERSON_RANGE_FOXGLOVE_TOPIC_WHITELIST,
                # список топіків, які взагалі видно у Foxglove — усе інше приховано
                'service_whitelist': ['(?!)'],
                # regex '(?!)' — це "негативний lookahead в порожньому місці",
                # він НІКОЛИ не збігається з жодним рядком. Тобто список
                # сервісів, дозволених для виклику через Foxglove — ПОРОЖНІЙ:
                # жодних сервісів ROS не можна викликати ззовні (безпека).
                'param_whitelist': ['(?!)'],
                # так само — жодні ROS-параметри не можна читати/міняти через Foxglove
                'client_topic_whitelist': ['(?!)'],
                # так само — клієнт (Foxglove Studio) не може публікувати
                # СВОЇ топіки назад у ROS-граф — тільки читає
                'capabilities': ['connectionGraph'],
                # які "можливості" протоколу foxglove_bridge увімкнені;
                # connectionGraph — дозволяє Foxglove показати граф нод/топіків
                'publish_client_count': True,
                # публікувати топік /foxglove_bridge/client_count — скільки
                # зараз підключено клієнтів (видно в списку whitelist вище)
            }],
            output='screen',
        ),

        # Mandatory local rqt image viewer (after cameras/topics come up).
        TimerAction(
            period=2.0,
            # period=2.0 — почекати 2.0 секунди ПІСЛЯ старту launch, і
            # тільки тоді виконати дії зі списку actions=[...] нижче.
            # Потрібно, бо камери й топіки з'являються не миттєво —
            # якщо відкрити вьювер одразу, він може нічого не знайти.
            actions=[Node(
                package='rqt_image_view',
                executable='rqt_image_view',
                name='person_range_rqt_image_view',
                arguments=[
                    LaunchConfiguration('image_view_topic'),
                    # який топік одразу відкрити у вьювері (з аргументу вище)
                    '--force-discover',
                    # прапорець самого rqt_image_view: примусово пересканувати
                    # список доступних топіків (інакше може не побачити нові)
                ],
                additional_env={'DISPLAY': LaunchConfiguration('display')},
                # передаємо змінну середовища DISPLAY, щоб GUI-вікно знало,
                # на якому екрані малюватись (X11)
                condition=IfCondition(rqt_image_view),
                # запускати тільки якщо launch-аргумент rqt_image_view=True
                output='screen',
            )],
        ),

        # Reliable OpenCV viewer (rqt_image_view needs SIP C++ plugins that may be broken).
        TimerAction(
            period=2.2,   # трохи пізніше за попередній вьювер, щоб не стартувати всі GUI одночасно
            actions=[Node(
                package='image_view',
                executable='image_view',
                name='person_range_image_view',
                remappings=[('image', '/stereo/left/image_rect')],
                # remappings — "перейменування" топіків для цієї конкретної
                # ноди: усередині ноди код підписаний на топік з іменем
                # 'image' (загальна, "родова" назва), а remapping каже:
                # "коли ця нода питає топік 'image', насправді підключи її
                # до реального топіка /stereo/left/image_rect". Це стандартний
                # спосіб переналаштувати готові (чужі) ноди без зміни їх коду.
                additional_env={'DISPLAY': LaunchConfiguration('display')},
                condition=IfCondition(image_view),
                output='screen',
            )],
        ),

        # Mandatory computation graph on Pi display.
        TimerAction(
            period=2.5,
            actions=[Node(
                package='rqt_graph',
                executable='rqt_graph',
                name='person_range_rqt_graph',
                arguments=['--force-discover'],
                additional_env={'DISPLAY': LaunchConfiguration('display')},
                condition=IfCondition(rqt_graph),
                output='screen',
            )],
        ),

        # Legacy heavy debug viewer (only with visualization:=true).
        TimerAction(
            period=3.0,
            actions=[Node(
                package='image_view',
                executable='image_view',
                name='person_range_view',
                remappings=[('image', '/person_range/debug_image')],
                # тут інший топік — важка debug-картинка з рамками, яку
                # публікує сама fusion-нода лише коли visualization=true
                additional_env={'DISPLAY': LaunchConfiguration('display')},
                condition=IfCondition(visualization),
                # запускається ТІЛЬКИ якщо явно передати visualization:=true
                output='screen',
            )],
        ),

    ])
    # Кінець списку дій → кінець LaunchDescription → кінець функції.
    # ros2 launch отримує цей об'єкт і виконує все, що в ньому описано.
