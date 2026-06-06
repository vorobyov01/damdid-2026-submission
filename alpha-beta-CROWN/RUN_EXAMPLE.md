# Как запустить простейший пример

## Шаг 1: Установка зависимостей

Сначала установите PyTorch (если еще не установлен):
```bash
pip3 install torch torchvision
```

## Шаг 2: Установка auto_LiRPA

```bash
cd /workspace/auto_LiRPA
pip3 install -e .
```

## Шаг 3: Запуск простейшего примера

Самый простой пример находится в `examples/simple/toy.py`:

```bash
cd /workspace/auto_LiRPA
python3 examples/simple/toy.py
```

Этот пример:
- Создает простую 2-слойную нейронную сеть
- Вычисляет границы выхода при заданных ограничениях на вход
- Демонстрирует различные методы вычисления границ (IBP, CROWN, alpha-CROWN)

## Альтернативный пример

Если хотите более продвинутый пример с реальной моделью MNIST:
```bash
python3 examples/vision/simple_verification.py
```
(Этот пример требует загрузки данных MNIST и предобученных весов)




