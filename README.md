<<<<<<< HEAD
# stereo-follow_me
=======
# camera_ws — **ПОВНИЙ** person-range пайплайн

> **Обов’язково:** для звичайної роботи, демо, перевірки на залізі та інтеграції  
> **завжди** запускайте **ПОВНИЙ** пайплайн. Часткові launch (лише stereo, лише YOLO)  
> — тільки для ізольованої розробки/тестів окремих пакетів, **не** як основний режим.

**Канонічний старт (єдиний правильний для runtime):**

```bash
cd ~/camera_ws
./run_person_range.sh
```

еквівалент:

```bash
source /opt/ros/jazzy/setup.bash
source ~/camera_ws/install/setup.bash
export DISPLAY=:0
ros2 launch person_range_fusion person_range.launch.py
```

**Не вимикайте** за замовчуванням увімкнені частини full stack:

| Компонент | Аргумент | Має бути |
|-----------|----------|----------|
| `foxglove_bridge` | `foxglove:=true` | **on**, порт **8765** |
| `rqt_image_view` | `rqt_image_view:=true` | **on** |
| `image_view` | `image_view:=true` | **on** |
| `rqt_graph` | `rqt_graph:=true` | **on** |

Якщо вимкнути bridge/GUI — це **неповний** пайплайн і для звичайної роботи **не використовується**.

---

## Що входить у ПОВНИЙ пайплайн

Стереокамери → глибина → YOLO (person) → fusion (Z) → **foxglove_bridge** + **rqt_image_view** + **image_view** + **rqt_graph**.

**Залізо:** Raspberry Pi 5 (типово 8 GB), дві камери IMX219, ROS 2 **Jazzy**.  
**Хост:** Mac / ПК з Foxglove Studio.  
**Локально на Pi:** дисплей (`DISPLAY=:0`).

```
ліва/права camera_ros
        ↓
  debayer + rectify
        ↓
     SGBM → /stereo/disparity
        ↓
  depth node → /stereo/depth          (метрична Z, 32FC1)
        ↘
         left image_rect ──→ YOLO (person, ~3 FPS)
              │                     ↓
              │              /person_detector/detections
              │                     ↓
              │                fusion (sync + ROI depth)
              │                     ↓
              │    annotations / scene / detections_3d / nearest_point
              │                     ↓
              │              ┌───────────┴───────────┐
              │              ▼                       ▼
              │     foxglove_bridge          rqt_image_view
              │        :8765                 + image_view
              │         │                    + rqt_graph
              │         │                    (DISPLAY=:0)
              │         │              topic: /stereo/left/image_rect
              │         └──→ Mac Foxglove Studio
              │
         pi_diagnostics → /person_range/diagnostics
```

Не піднімайте **два** екземпляри full launch одночасно.

---

## Швидкий старт

```bash
cd ~/camera_ws
./run_person_range.sh
```

Скрипт `run_person_range.sh`:

- перевіряє ROS 2 Jazzy і `install/setup.bash`
- виставляє `DISPLAY=:0`
- зупиняє старий `person-range-combined.service` (щоб не було двох стеків)
- запускає `ros2 launch person_range_fusion person_range.launch.py`

**Зупинка:** `Ctrl+C` у тому ж терміналі.

**Foxglove на Mac/ПК:** bridge слухає **усі мережі** (`0.0.0.0:8765`).

| URL | Коли працює |
|-----|-------------|
| **`ws://qqqq:8765`** | Клієнт у **Wi‑Fi hotspot Pi** (DNS роздає dnsmasq: `qqqq` → `192.168.1.1`) |
| `ws://qqqq.local:8765` | mDNS/Avahi (hotspot або LAN, Mac/Linux) |
| `ws://192.168.1.1:8765` | hotspot, завжди за IP |
| `ws://192.168.0.x:8765` | eth0 LAN |

Коротке ім’я **`qqqq` без `.local`** — через DHCP DNS хотспота
(`/etc/NetworkManager/dnsmasq-shared.d/qqqq.conf`). На eth0 LAN
коротке `qqqq` **не** резолвиться (лише IP або `.local`).

Layout: `foxglove/person_range_clean.json`.

Додаткові аргументи (лише якщо потрібно; **не** вимикайте full stack без причини):

```bash
./run_person_range.sh visualization:=true   # важкий debug_image (опційно)
```

---

## 1. Одноразова підготовка

### 1.1. ROS 2 і workspace

```bash
source /opt/ros/jazzy/setup.bash
cd ~/camera_ws
colcon build --symlink-install
```

### 1.2. YOLO (NCNN) і venv

| Що | Шлях |
|----|------|
| Python | `~/.venvs/camera-yolo/bin/python` |
| Модель | `~/.local/share/camera-yolo/models/yolo26n_ncnn_model/` |

Інші шляхи: launch-аргументи `python_executable` і `model_path` (див. §5).

### 1.3. Калібрування стерео

Для профілю `person_range` (320×240) потрібні YAML live 320×240. Без них rectify/disparity некоректні.

### 1.4. Foxglove Studio

Layout: `~/camera_ws/foxglove/person_range_clean.json`

### 1.5. GUI на Pi

```bash
sudo apt install ros-jazzy-rqt-image-view
```

Потрібна графічна сесія та `DISPLAY` (типово `:0`).

---

## 2. Запуск ПОВНОГО пайплайну (єдиний runtime-режим)

На **Raspberry Pi**:

```bash
source /opt/ros/jazzy/setup.bash
source ~/camera_ws/install/setup.bash
cd ~/camera_ws
export DISPLAY=:0
systemctl --user stop person-range-combined.service 2>/dev/null || true
ros2 launch person_range_fusion person_range.launch.py
```

За замовчуванням (повний stack):

- **`foxglove:=true`** — bridge **0.0.0.0:8765**
- **`rqt_image_view:=true`** — rqt на `/stereo/left/image_rect`
- **`image_view:=true`** — OpenCV fallback
- **`rqt_graph:=true`** — граф нод
- `visualization:=false` — без важкого `debug_image`

GUI відкриваються через ~2–3 с. Залиште термінал відкритим.

### Автозапуск при старті Raspberry Pi (systemd user)

Повний пайплайн (`./run_person_range.sh`) як **user service** з linger
(старт без ручного логіну):

```bash
cd ~/camera_ws
./scripts/install_person_range_autostart.sh
```

Це:

1. ставить unit `~/.config/systemd/user/person-range-combined.service`
2. вмикає **linger** для користувача (`loginctl enable-linger`)
3. `enable` + `restart` сервісу

| Команда | Дія |
|---------|-----|
| `systemctl --user status person-range-combined` | статус |
| `journalctl --user -u person-range-combined -f` | логи |
| `systemctl --user stop person-range-combined` | стоп |
| `systemctl --user start person-range-combined` | старт |
| `systemctl --user disable --now person-range-combined` | вимкнути автозапуск |

Unit чекає до ~2 хв на X11 (`DISPLAY=:0`) для `image_view`/`rqt`, потім
все одно стартує (foxglove/камери не залежать від GUI).

Лише enable без негайного restart: `./scripts/install_person_range_autostart.sh --no-start`.

---

## 3. Перевірка, що full stack живий

```bash
source /opt/ros/jazzy/setup.bash
source ~/camera_ws/install/setup.bash
ros2 node list
```

Очікувано:

- `/stereo/left/camera_node`, `/stereo/right/camera_node`
- debayer / rectify left & right
- `/stereo/disparity_node`
- `/stereo/depth`
- `/person_detector/detector`
- `/person_range/fusion`
- `/person_range/pi_diagnostics`
- **`/foxglove_bridge`**
- **`/person_range_rqt_image_view`** (або `/rqt_gui_py_node_…`)
- **`/person_range_image_view`**, **`/person_range_rqt_graph`**

```bash
ros2 topic hz /stereo/left/image_rect      # ~12 Hz
ros2 topic hz /stereo/disparity            # ~12 Hz
ros2 topic hz /stereo/depth                # ~12 Hz
ros2 topic hz /person_detector/detections  # ~3 Hz
ros2 topic hz /person_range/image_annotations
ros2 topic hz /person_range/diagnostics    # ~1 Hz

# Hotspot AP (клієнти на Wi‑Fi Pi):
ip -4 -o addr show wlan0 | awk '{print $4}'
# Усі IPv4:
hostname -I
ss -ltn | grep 8765   # очікувано 0.0.0.0:8765
```

---

## 4. Foxglove (Mac / ПК)

Bridge = **`0.0.0.0:8765`** (доступ з **будь-якої** мережі Pi: hotspot, eth0, …).

1. **Open connection** → Foxglove WebSocket:
   - **Hotspot (рекомендовано):** `ws://qqqq:8765` — коротке ім’я **без**
     `.local` (DNS хотспота Pi)
   - або mDNS: `ws://qqqq.local:8765`
   - або IP: `ws://192.168.1.1:8765` (hotspot) / eth0-IP (LAN)
2. **Layout → Import** → `foxglove/person_range_clean.json`

Якщо `qqqq` не відкривається: переконайтесь, що Mac/ПК у Wi‑Fi
**Pi-Camera** (hotspot) і бере DNS з Pi; fallback — IP або `.local`.

| Панель | Дані |
|--------|------|
| Image | ліва камера + annotations |
| 3D | `/stereo/depth` (optical) + кубики людей (`stereo_left_up`) |
| Diagnostics | temp, power, RSS |
| Plot | X forward (дальність, REP-103) |

**Fixed frame / followTf:** `stereo_left_up` (REP-103: X-вперед, Y-вліво, Z-вгору).
Хмара з depth лишається в optical через `camera_info`; кубики — у `stereo_left_up`;
їх зв’язує static TF. Не крутіть розмір куба вручну — `person_marker_size=[0.5,0.5,1.7]`.

Якщо 3D порожній: для `/stereo/depth` → **Camera info** = `/stereo/left/camera_info`.

---

## 5. Аргументи launch

Повний пайплайн **зберігає** foxglove/rqt/image_view/rqt_graph увімкненими:

```bash
ros2 launch person_range_fusion person_range.launch.py \
  foxglove:=true \
  rqt_image_view:=true \
  image_view:=true \
  rqt_graph:=true \
  image_view_topic:=/stereo/left/image_rect \
  display:=:0 \
  visualization:=false \
  yolo_fps:=3.0 \
  confidence_threshold:=0.40 \
  imgsz:=320
```

| Аргумент | За замовчуванням | Опис |
|----------|------------------|------|
| `foxglove` | **`true`** | bridge на 8765 (**обов’язково** у full pipeline) |
| `rqt_image_view` | **`true`** | rqt viewer на Pi (**обов’язково**) |
| `image_view` | **`true`** | OpenCV viewer (**обов’язково**) |
| `rqt_graph` | **`true`** | граф нод (**обов’язково**) |
| `image_view_topic` | `/stereo/left/image_rect` | топік для viewer |
| `display` | `:0` | X11 `DISPLAY` |
| `visualization` | `false` | важкий `debug_image` (опційно) |
| `yolo_fps` | `3.0` | ліміт інференсу |
| `confidence_threshold` | `0.40` | поріг детекції |
| `imgsz` | `320` | вхід YOLO |
| `model_path` | `~/.../yolo26n_ncnn_model` | NCNN |
| `python_executable` | `~/.venvs/camera-yolo/bin/python` | venv |

Опційний важкий debug (full stack лишається):

```bash
./run_person_range.sh visualization:=true
```

---

## 6. Зупинка

```bash
# Ctrl+C у терміналі launch
# або:
systemctl --user stop person-range-combined.service
```

---

## 7. Типові проблеми

| Симптом | Що перевірити |
|---------|----------------|
| Немає зображень | камери, логи `camera_node` |
| Disparity/depth 0 Hz | калібрування 320×240, обидві камери |
| YOLO не стартує | `model_path`, venv ultralytics+ncnn |
| Foxglove не бачить Pi | `foxglove:=true`; `ss -ltn \| grep 8765` → `0.0.0.0:8765`; спробуйте `ws://qqqq.local:8765` або IP інтерфейсу клієнта; mDNS: `systemctl status avahi-daemon` |
| rqt не відкривається | `DISPLAY=:0`, GUI-сесія |
| Другий launch ламає graph | зупинити перший full stack |
| `throttled` | БЖ Pi 5 (5 V / 5 A) |

---

## 8. Шпаргалка

```bash
# ПОВНИЙ пайплайн — завжди так для runtime
cd ~/camera_ws && ./run_person_range.sh

# IP / domain / bridge
# hotspot DNS: ws://qqqq:8765   |  mDNS: ws://qqqq.local:8765
hostname -I
ss -ltn | grep 8765        # → 0.0.0.0:8765

# Mac у hotspot Pi-Camera → Foxglove:
# ws://qqqq:8765  +  foxglove/person_range_clean.json

# Стоп
# Ctrl+C  або  systemctl --user stop person-range-combined.service
```

---

## 9. Файли

| Шлях | Опис |
|------|------|
| `run_person_range.sh` | **Єдиний** one-shot старт full pipeline |
| `systemd/person-range-combined.service` | User unit: автозапуск full pipeline |
| `scripts/install_person_range_autostart.sh` | Install + enable linger + enable unit |
| `src/person_range_fusion/launch/person_range.launch.py` | Combined launch (full stack) |
| `src/person_range_fusion/README.md` | Топіки, параметри fusion |
| `AGENTS.md` | Правила для агентів: завжди full pipeline |
| `foxglove/person_range_clean.json` | Layout Foxglove |

### Лише для ізольованої розробки (НЕ runtime)

```bash
# не замінює full pipeline
ros2 launch drone_stereo_bringup stereo_sgbm_light.launch.py profile:=person_range
ros2 launch yolo_person_car person_test.launch.py
```
>>>>>>> 8368694 (Initial import: stereo follow_me pipeline from RPi5)
